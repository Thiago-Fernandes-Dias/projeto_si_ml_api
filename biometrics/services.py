import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from django.conf import settings
from django.db import models, transaction

from biometrics.ml.features import (
    FeatureValidationError,
    build_schema,
    features_to_positional_vector,
    mean_hold_time_ms,
    positional_schema,
    validate_vector,
)
from biometrics.ml.inference import classify
from biometrics.ml.m2005 import AdaptiveM2005Alg
from biometrics.ml.registry import load_model, save_model
from biometrics.ml.training import adapt_user_model, get_m2005_config, train_user_model
from biometrics.models import (
    Amostra,
    DecisaoBiometrica,
    ModeloBiometrico,
    OrigemAmostra,
    SampleLabel,
    TrainingTrigger,
    User,
)


class UserNotFoundError(Exception):
    """Raised when the target username does not exist."""


class AlreadyEnrolledError(Exception):
    """Raised when a user attempts enrollment more than once for the current password."""


class NotEnrolledError(Exception):
    """Raised when a user attempts biometric authentication before completing enrollment."""


class EnrollmentValidationError(ValueError):
    """Raised when enrollment payload violates sample count or feature schema constraints."""


class ModelVersionNotFoundError(Exception):
    """Raised when a requested model version does not exist for the user."""


@dataclass(frozen=True)
class AuthenticationOutcome:
    decision: str
    confidence: float
    threshold: float
    inference_time_ms: float
    mean_hold_time_ms: float
    user_average_hold_time_ms: float
    model_id: int
    model_version: int
    sample_label: str
    should_record_for_retraining: bool
    message: str


def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def _resolve_user(
    username: str | None = None,
    user_id: str | uuid.UUID | None = None,
) -> User | None:
    if user_id is not None:
        try:
            uid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
            found = User.objects.filter(id=uid).first()
            if found is not None:
                return found
        except (ValueError, TypeError):
            pass

    if username:
        clean = normalize_username(username)
        return User.objects.filter(nome__iexact=clean).first()
    return None


def get_active_model(user: User) -> ModeloBiometrico | None:
    models_qs = ModeloBiometrico.objects.filter(
        usuario=user,
        criado_em__gte=user.data_mudanca_senha,
    ).order_by("-versao", "-id")
    first_model = models_qs.first()
    if first_model is None:
        first_model = (
            ModeloBiometrico.objects.filter(usuario=user).order_by("-versao", "-id").first()
        )
    if first_model is None:
        return None

    if first_model.caminho_arquivo:
        active_file = Path(first_model.caminho_arquivo).parent / "active_version.json"
        if active_file.is_file():
            try:
                data = json.loads(active_file.read_text(encoding="utf-8"))
                active_v = data.get("active_version")
                if isinstance(active_v, int):
                    chosen = ModeloBiometrico.objects.filter(
                        usuario=user,
                        versao=active_v,
                    ).first()
                    if chosen is not None:
                        return chosen
            except Exception:
                pass
    return first_model


def _extract_vectors_from_payload_samples(
    samples: Sequence[Any],
) -> tuple[list[np.ndarray], list[float], list[str] | None]:
    extracted_vectors: list[np.ndarray] = []
    durations_ms: list[float] = []
    feature_dicts: list[Mapping[str, float]] = []

    for idx, item in enumerate(samples, start=1):
        if isinstance(item, Sequence) and not isinstance(item, str | bytes | Mapping):
            vec = validate_vector(item)
            extracted_vectors.append(vec)
            durations_ms.append(float(np.sum(vec[0::3]) * 1000.0))
            continue

        if not isinstance(item, Mapping):
            raise EnrollmentValidationError(f"Amostra #{idx} tem formato inválido.")

        duration = float(item.get("totalDurationMs", item.get("duracao_total_ms", 0.0)) or 0.0)

        if "vector" in item or "vetor" in item:
            raw_vec = item.get("vector", item.get("vetor"))
            vec = validate_vector(raw_vec)
            extracted_vectors.append(vec)
            durations_ms.append(duration or float(np.sum(vec[0::3]) * 1000.0))
        elif "features" in item and isinstance(item["features"], Mapping):
            feature_dicts.append(item["features"])
            durations_ms.append(duration)
        else:
            feature_dicts.append(item)
            durations_ms.append(duration)

    dict_schema: list[str] | None = None
    if feature_dicts:
        if extracted_vectors:
            raise EnrollmentValidationError(
                "Não é permitido misturar amostras por 'features' e por 'vector' no mesmo lote."
            )
        dict_schema = build_schema(feature_dicts)
        for feat in feature_dicts:
            vec = features_to_positional_vector(feat, schema=dict_schema)
            extracted_vectors.append(vec)

    if not extracted_vectors:
        raise EnrollmentValidationError("Nenhum vetor biométrico válido encontrado nas amostras.")

    first_len = len(extracted_vectors[0])
    for idx, vec in enumerate(extracted_vectors[1:], start=2):
        if len(vec) != first_len:
            raise EnrollmentValidationError(
                f"Amostra #{idx} possui dimensão {len(vec)}, diferente da amostra #1 ({first_len})."
            )

    return extracted_vectors, durations_ms, dict_schema


def enroll_user(
    username: str | None = None,
    samples: Sequence[Any] | None = None,
    samples_count: int | None = None,
    user_id: str | uuid.UUID | None = None,
) -> ModeloBiometrico:
    user = _resolve_user(username=username, user_id=user_id)
    if user is None:
        raise UserNotFoundError(f"Usuário '{username or user_id}' não encontrado.")

    if ModeloBiometrico.objects.filter(
        usuario=user,
        criado_em__gte=user.data_mudanca_senha,
    ).exists():
        raise AlreadyEnrolledError(f"O usuário '{user.nome}' já realizou o enrollment.")

    expected_count = int(settings.BIOMETRICS["ENROLLMENT_SAMPLES"])

    if samples is None or len(samples) == 0:
        db_samples = list(
            Amostra.objects.filter(
                usuario=user,
                origem=OrigemAmostra.CADASTRO,
                coletada_em__gte=user.data_mudanca_senha,
            ).order_by("repeticao", "id")[:expected_count]
        )
        if len(db_samples) != expected_count:
            raise EnrollmentValidationError(
                f"O enrollment exige exatamente {expected_count} amostras de 'Cadastro' "
                f"(encontradas no banco: {len(db_samples)})."
            )
        try:
            vectors = [validate_vector(s.vetor) for s in db_samples]
        except FeatureValidationError as exc:
            raise EnrollmentValidationError(str(exc)) from exc

        first_len = len(vectors[0])
        for idx, vec in enumerate(vectors[1:], start=2):
            if len(vec) != first_len:
                raise EnrollmentValidationError(
                    f"Amostra de cadastro #{idx} no banco possui tamanho {len(vec)} != {first_len}."
                )
        dict_schema = None
    else:
        actual_len = len(samples) if isinstance(samples, Sequence) else 0
        if not isinstance(samples, Sequence) or actual_len != expected_count:
            raise EnrollmentValidationError(
                f"O enrollment exige exatamente {expected_count} amostras (recebido: {actual_len})."
            )
        if samples_count is not None and int(samples_count) != len(samples):
            raise EnrollmentValidationError(
                f"Inconsistência entre samplesCount ({samples_count}) "
                f"e o tamanho de samples ({len(samples)})."
            )
        try:
            vectors, _, dict_schema = _extract_vectors_from_payload_samples(samples)
        except FeatureValidationError as exc:
            raise EnrollmentValidationError(str(exc)) from exc

    genuine_matrix = np.vstack(vectors)
    pos_schema = positional_schema(genuine_matrix.shape[1])

    training_result = train_user_model(
        genuine=genuine_matrix,
        schema=pos_schema,
    )
    if dict_schema is not None:
        training_result.model.source_feature_keys_ = dict_schema  # type: ignore[attr-defined]

    with transaction.atomic():
        locked_user = User.objects.select_for_update().get(id=user.id)
        if ModeloBiometrico.objects.filter(
            usuario=locked_user,
            criado_em__gte=locked_user.data_mudanca_senha,
        ).exists():
            raise AlreadyEnrolledError(f"O usuário '{locked_user.nome}' já realizou o enrollment.")

        trained_model = save_model(
            user=locked_user,
            result=training_result,
            trigger=TrainingTrigger.ENROLLMENT,
        )

    return trained_model


def label_for(
    probability: float,
    threshold: float | None = None,
    margin: float | None = None,
) -> SampleLabel:
    decision_threshold = (
        float(settings.BIOMETRICS["DECISION_THRESHOLD"]) if threshold is None else float(threshold)
    )
    confidence_margin = (
        float(settings.BIOMETRICS["LABEL_CONFIDENCE_MARGIN"]) if margin is None else float(margin)
    )

    if probability >= decision_threshold + confidence_margin:
        return SampleLabel.GENUINE
    if probability <= decision_threshold - confidence_margin:
        return SampleLabel.IMPOSTOR
    return SampleLabel.UNCERTAIN


def authenticate_user(
    username: str | None = None,
    features: Mapping[str, float] | None = None,
    vector: Sequence[float] | None = None,
    user_id: str | uuid.UUID | None = None,
) -> AuthenticationOutcome:
    user = _resolve_user(username=username, user_id=user_id)
    if user is None:
        raise UserNotFoundError(f"Usuário '{username or user_id}' não encontrado.")

    active_model = get_active_model(user)
    if active_model is None:
        raise NotEnrolledError(
            "Usuário não possui modelo biométrico ativo. Conclua o enrollment primeiro."
        )

    estimator = load_model(active_model)
    expected_dim = len(getattr(estimator, "feature_names_", []))

    if vector is not None:
        pos_vector = validate_vector(vector)
    elif features is not None:
        source_keys = getattr(estimator, "source_feature_keys_", None)
        pos_vector = features_to_positional_vector(features, schema=source_keys)
    else:
        raise FeatureValidationError(
            "Nenhuma amostra biométrica (vector ou features) informada."
        )

    if expected_dim > 0 and len(pos_vector) != expected_dim:
        raise FeatureValidationError(
            f"Dimensão do vetor ({len(pos_vector)}) incompatível com o modelo ({expected_dim})."
        )

    threshold = float(getattr(estimator, "threshold", settings.BIOMETRICS["DECISION_THRESHOLD"]))
    classification = classify(estimator, pos_vector, threshold=threshold)
    sample_label = label_for(classification.genuine_probability, threshold=threshold)
    should_record_for_retraining = bool(
        classification.decision == "GENUINE" and sample_label == SampleLabel.GENUINE
    )

    attempt_hold_ms = mean_hold_time_ms(pos_vector)
    gallery = getattr(estimator, "_gallery", [])
    if gallery:
        user_hold_ms = float(
            np.mean([mean_hold_time_ms(sample_row.to_numpy()) for sample_row in gallery])
        )
    else:
        user_hold_ms = attempt_hold_ms

    if classification.decision == "GENUINE":
        message = "Padrão biométrico compatível com o perfil legítimo cadastrado."
    else:
        message = "Acesso bloqueado: o ritmo de digitação diverge do perfil legítimo."

    return AuthenticationOutcome(
        decision=classification.decision,
        confidence=classification.genuine_probability,
        threshold=threshold,
        inference_time_ms=classification.inference_time_ms,
        mean_hold_time_ms=attempt_hold_ms,
        user_average_hold_time_ms=user_hold_ms,
        model_id=int(active_model.id),
        model_version=int(active_model.versao),
        sample_label=str(sample_label),
        should_record_for_retraining=should_record_for_retraining,
        message=message,
    )


def retrain_user(
    user: User,
    trigger: TrainingTrigger | str = TrainingTrigger.SCHEDULED,
    force: bool = False,
) -> ModeloBiometrico | None:
    with transaction.atomic():
        locked_user = User.objects.select_for_update(skip_locked=True).filter(id=user.id).first()
        if locked_user is None:
            return None

        active_model = get_active_model(locked_user)
        if active_model is None and not Amostra.objects.filter(
            usuario=locked_user,
            origem=OrigemAmostra.CADASTRO,
            coletada_em__gte=locked_user.data_mudanca_senha,
        ).exists():
            return None

        base_threshold = float(settings.BIOMETRICS["DECISION_THRESHOLD"])
        prev_estimator: Any = None
        if active_model is not None:
            try:
                prev_estimator = load_model(active_model)
                base_threshold = float(getattr(prev_estimator, "threshold", base_threshold))
            except Exception:
                prev_estimator = None
        margin = float(settings.BIOMETRICS["LABEL_CONFIDENCE_MARGIN"])
        genuine_min_conf = base_threshold + margin

        new_logins_qs = Amostra.objects.filter(
            usuario=locked_user,
            origem=OrigemAmostra.LOGIN,
            decisao=DecisaoBiometrica.GENUINE,
            confianca__gte=genuine_min_conf,
            coletada_em__gte=locked_user.data_mudanca_senha,
        )
        if active_model is not None:
            new_logins_qs = new_logins_qs.filter(coletada_em__gt=active_model.criado_em)

        min_new_samples = int(settings.BIOMETRICS["RETRAIN_MIN_NEW_SAMPLES"])
        if not force and active_model is not None and new_logins_qs.count() < min_new_samples:
            return None

        cfg = get_m2005_config()
        can_adapt_incrementally = (
            active_model is not None
            and active_model.criado_em >= locked_user.data_mudanca_senha
            and isinstance(prev_estimator, AdaptiveM2005Alg)
            and cfg.algorithm in {"M2005_IDB", "M2005_DB"}
            and prev_estimator.variant == cfg.variant
        )

        if can_adapt_incrementally:
            new_samples = list(new_logins_qs.order_by("coletada_em", "id"))
            expected_dim = len(getattr(prev_estimator, "feature_names_", []))
            new_vectors = [
                np.asarray(s.vetor, dtype=np.float64)
                for s in new_samples
                if expected_dim == 0 or len(s.vetor) == expected_dim
            ]
            new_matrix = np.vstack(new_vectors) if new_vectors else None
            training_result = adapt_user_model(
                prev_estimator,
                new_genuine=new_matrix,
                schema=list(getattr(prev_estimator, "feature_names_", [])) or None,
            )
        else:
            genuine_window = int(settings.BIOMETRICS["RETRAIN_GENUINE_WINDOW"])
            genuine_samples = list(
                Amostra.objects.filter(
                    usuario=locked_user,
                    coletada_em__gte=locked_user.data_mudanca_senha,
                )
                .filter(
                    models.Q(origem=OrigemAmostra.CADASTRO)
                    | models.Q(
                        origem=OrigemAmostra.LOGIN,
                        decisao=DecisaoBiometrica.GENUINE,
                        confianca__gte=genuine_min_conf,
                    )
                )
                .order_by("coletada_em", "id")
            )
            if not genuine_samples:
                prev_gallery = getattr(prev_estimator, "_gallery", [])
                if not prev_gallery:
                    return None
                genuine_vectors = [
                    row.to_numpy(dtype=np.float64) for row in prev_gallery
                ]
                expected_dim = len(genuine_vectors[0])
            else:
                if cfg.algorithm not in {"M2005_IDB", "M2005_DB"}:
                    genuine_samples = genuine_samples[-genuine_window:]
                expected_dim = len(genuine_samples[0].vetor)
                genuine_vectors = [
                    np.asarray(s.vetor, dtype=np.float64)
                    for s in genuine_samples
                    if len(s.vetor) == expected_dim
                ]
            if not genuine_vectors:
                return None

            genuine_matrix = np.vstack(genuine_vectors)
            schema = (
                list(getattr(prev_estimator, "feature_names_", []))
                if prev_estimator is not None
                and len(getattr(prev_estimator, "feature_names_", [])) == expected_dim
                else positional_schema(expected_dim)
            )
            training_result = train_user_model(
                genuine=genuine_matrix,
                schema=schema,
            )
            if prev_estimator is not None:
                prev_keys = getattr(prev_estimator, "source_feature_keys_", None)
                if prev_keys is not None:
                    training_result.model.source_feature_keys_ = prev_keys  # type: ignore[attr-defined]

        return save_model(user=locked_user, result=training_result, trigger=trigger)


def retrain_due_users(now: Any = None, force: bool = False) -> list[ModeloBiometrico]:
    retrained: list[ModeloBiometrico] = []
    users = (
        User.objects.filter(
            models.Q(biometria_cadastrada_em__isnull=False) | models.Q(modelos__isnull=False)
        )
        .distinct()
        .order_by("criado_em", "id")
    )

    for user in users:
        new_model = retrain_user(
            user=user,
            trigger=TrainingTrigger.SCHEDULED,
            force=force,
        )
        if new_model is not None:
            retrained.append(new_model)
    return retrained


def activate_model_version(username: str, version: int) -> ModeloBiometrico:
    user = _resolve_user(username=username)
    if user is None:
        raise UserNotFoundError(f"Usuário '{username}' não encontrado.")

    target = ModeloBiometrico.objects.filter(usuario=user, versao=version).first()
    if target is None:
        raise ModelVersionNotFoundError(
            f"Versão v{version} não encontrada para o usuário '{user.nome}'."
        )

    load_model(target)

    if target.caminho_arquivo:
        active_file = Path(target.caminho_arquivo).parent / "active_version.json"
        active_file.parent.mkdir(parents=True, exist_ok=True)
        active_file.write_text(
            json.dumps({"active_version": int(version)}, indent=2),
            encoding="utf-8",
        )

    return target
