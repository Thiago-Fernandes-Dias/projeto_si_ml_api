import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from django.conf import settings
from django.db import models, transaction

from biometrics.logging_utils import log_structured
from biometrics.ml.features import (
    FeatureValidationError,
    mean_hold_time_ms,
    positional_schema,
    validate_vector,
)
from biometrics.ml.inference import classify
from biometrics.ml.m2005 import AdaptiveM2005Alg
from biometrics.ml.registry import clear_model_cache, load_model, save_model, write_text_atomic
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

logger = logging.getLogger(__name__)


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
class ModelRollbackOutcome:
    deleted_count: int
    active_version: int | None


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
                log_structured(
                    logger,
                    logging.DEBUG,
                    "ml_api.user.resolved",
                    "Usuário resolvido por userId.",
                    lookupBy="userId",
                    found=True,
                    userId=str(found.id),
                    username=found.nome,
                )
                return found
        except (ValueError, TypeError):
            log_structured(
                logger,
                logging.DEBUG,
                "ml_api.user.invalid_id",
                "Formato inválido de userId recebido na ML API.",
                userId=str(user_id),
            )

    if username:
        clean = normalize_username(username)
        found = User.objects.filter(nome__iexact=clean).first()
        log_structured(
            logger,
            logging.DEBUG,
            "ml_api.user.resolved",
            "Consulta de usuário por username concluída.",
            lookupBy="username",
            found=found is not None,
            username=clean,
            userId=str(found.id) if found is not None else None,
        )
        return found
    return None


def get_active_model(user: User) -> ModeloBiometrico | None:
    models_qs = ModeloBiometrico.objects.filter(
        usuario=user,
        criado_em__gte=user.data_mudanca_senha,
    ).order_by("-versao", "-id")
    first_model = models_qs.first()
    if first_model is None:
        return None

    if first_model.caminho_arquivo:
        active_file = Path(first_model.caminho_arquivo).parent / "active_version.json"
        if active_file.is_file():
            try:
                data = json.loads(active_file.read_text(encoding="utf-8"))
                active_v = data.get("active_version")
                if isinstance(active_v, int):
                    chosen = models_qs.filter(versao=active_v).first()
                    if chosen is not None:
                        return chosen
            except Exception:
                pass
    return first_model


def enroll_user(
    username: str | None = None,
    vectors: Sequence[Any] | None = None,
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

    received_count = 0
    if isinstance(vectors, Sequence) and not isinstance(vectors, str | bytes | Mapping):
        received_count = len(vectors)

    if received_count != expected_count:
        raise EnrollmentValidationError(
            f"O enrollment exige exatamente {expected_count} vetores (recebido: {received_count})."
        )

    try:
        validated_vectors = [validate_vector(vector) for vector in vectors]
    except FeatureValidationError as exc:
        raise EnrollmentValidationError(str(exc)) from exc

    first_len = len(validated_vectors[0])
    for idx, vector in enumerate(validated_vectors[1:], start=2):
        if len(vector) != first_len:
            raise EnrollmentValidationError(
                f"Vetor #{idx} possui dimensão {len(vector)}, diferente do vetor #1 ({first_len})."
            )

    genuine_matrix = np.vstack(validated_vectors)
    pos_schema = positional_schema(genuine_matrix.shape[1])

    log_structured(
        logger,
        logging.DEBUG,
        "ml_api.enrollment.training_started",
        "Vetores preparados para treinamento inicial de enrollment.",
        flow="enrollment",
        userId=str(user.id),
        username=user.nome,
        sampleCount=int(genuine_matrix.shape[0]),
        vectorDimension=int(genuine_matrix.shape[1]),
        inputSource="backend_vectors",
    )

    training_result = train_user_model(
        genuine=genuine_matrix,
        schema=pos_schema,
    )

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

    log_structured(
        logger,
        logging.DEBUG,
        "ml_api.enrollment.completed",
        "Modelo inicial persistido após enrollment.",
        flow="enrollment",
        userId=str(locked_user.id),
        username=locked_user.nome,
        modelId=trained_model.id,
        modelVersion=trained_model.versao,
        algorithm=trained_model.algoritmo,
        threshold=training_result.stats.get("threshold"),
    )
    return trained_model


def label_for(
    probability: float,
    threshold: float | None = None,
) -> SampleLabel:
    decision_threshold = (
        float(settings.BIOMETRICS["DECISION_THRESHOLD"]) if threshold is None else float(threshold)
    )
    return SampleLabel.GENUINE if probability >= decision_threshold else SampleLabel.IMPOSTOR


def authenticate_user(
    username: str | None = None,
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

    if vector is None:
        raise FeatureValidationError("Nenhum vetor biométrico informado.")

    pos_vector = validate_vector(vector)

    if expected_dim > 0 and len(pos_vector) != expected_dim:
        raise FeatureValidationError(
            f"Dimensão do vetor ({len(pos_vector)}) incompatível com o modelo ({expected_dim})."
        )

    threshold = float(getattr(estimator, "threshold", settings.BIOMETRICS["DECISION_THRESHOLD"]))
    log_structured(
        logger,
        logging.DEBUG,
        "ml_api.login.inference_starting",
        "Executando inferência biométrica.",
        flow="login",
        userId=str(user.id),
        username=user.nome,
        modelId=int(active_model.id),
        modelVersion=int(active_model.versao),
        algorithm=active_model.algoritmo,
        inputType="vector",
        vectorDimension=int(len(pos_vector)),
        threshold=round(threshold, 4),
    )
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

    log_structured(
        logger,
        logging.DEBUG,
        "ml_api.login.inference_completed",
        "Resultado da inferência biométrica calculado.",
        flow="login",
        userId=str(user.id),
        username=user.nome,
        modelId=int(active_model.id),
        modelVersion=int(active_model.versao),
        decision=classification.decision,
        confidence=round(classification.genuine_probability, 4),
        threshold=round(threshold, 4),
        sampleLabel=str(sample_label),
        shouldRecordForRetraining=should_record_for_retraining,
        inferenceTimeMs=round(classification.inference_time_ms, 3),
        attemptHoldMs=round(attempt_hold_ms, 2),
        userHoldMs=round(user_hold_ms, 2),
    )
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

        retrain_threshold = float(settings.BIOMETRICS["DECISION_THRESHOLD"])
        prev_estimator: Any = None
        if active_model is not None:
            try:
                prev_estimator = load_model(active_model)
                retrain_threshold = float(
                    getattr(prev_estimator, "threshold", retrain_threshold)
                )
            except Exception:
                prev_estimator = None
        new_logins_qs = Amostra.objects.filter(
            usuario=locked_user,
            origem=OrigemAmostra.LOGIN,
            decisao=DecisaoBiometrica.GENUINE,
            confianca__gte=retrain_threshold,
            coletada_em__gte=locked_user.data_mudanca_senha,
        )
        if active_model is not None:
            new_logins_qs = new_logins_qs.filter(coletada_em__gt=active_model.criado_em)

        min_new_samples = int(settings.BIOMETRICS["RETRAIN_MIN_NEW_SAMPLES"])
        new_logins_count = new_logins_qs.count()
        if not force and active_model is not None and new_logins_count < min_new_samples:
            log_structured(
                logger,
                logging.DEBUG,
                "ml_api.retrain.skipped",
                "Retreinamento adiado: amostras novas insuficientes.",
                flow="retrain",
                userId=str(locked_user.id),
                username=locked_user.nome,
                newGenuineSamples=new_logins_count,
                minRequired=min_new_samples,
            )
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
                        confianca__gte=retrain_threshold,
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

        saved = save_model(user=locked_user, result=training_result, trigger=trigger)
        log_structured(
            logger,
            logging.DEBUG,
            "ml_api.retrain.completed",
            "Nova versão do modelo salva após retreinamento.",
            flow="retrain",
            userId=str(locked_user.id),
            username=locked_user.nome,
            mode="incremental" if can_adapt_incrementally else "full_window",
            modelId=saved.id,
            modelVersion=saved.versao,
            trigger=str(trigger),
        )
        return saved


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


def rollback_models_after(
    created_after: datetime,
    username: str | None = None,
    user_id: str | uuid.UUID | None = None,
) -> ModelRollbackOutcome:
    """Removes model versions created after a password change timestamp."""
    user = _resolve_user(username=username, user_id=user_id)
    if user is None:
        raise UserNotFoundError(f"Usuário '{username or user_id}' não encontrado.")

    models = list(
        ModeloBiometrico.objects.filter(
            usuario=user,
            criado_em__gte=created_after,
        )
    )
    paths = [Path(model.caminho_arquivo) for model in models if model.caminho_arquivo]

    with transaction.atomic():
        for model in models:
            model.delete()

        remaining = (
            ModeloBiometrico.objects.filter(usuario=user)
            .order_by("-versao", "-id")
            .first()
        )
        if remaining is not None and remaining.caminho_arquivo:
            active_file = Path(remaining.caminho_arquivo).parent / "active_version.json"
        else:
            models_root = Path(settings.BIOMETRICS["MODELS_DIR"])
            active_file = models_root / str(user.id) / "active_version.json"

        if remaining is not None:
            write_text_atomic(
                active_file,
                json.dumps({"active_version": int(remaining.versao)}, indent=2),
            )
        else:
            active_file.unlink(missing_ok=True)

    clear_model_cache()

    for path in paths:
        for candidate in (path, Path(f"{path}.meta.json")):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                logger.warning("Não foi possível remover artefato de modelo '%s'.", candidate)

    active_version = int(remaining.versao) if remaining is not None else None
    log_structured(
        logger,
        logging.DEBUG,
        "ml_api.models.rolled_back",
        "Modelos posteriores à troca de senha removidos.",
        flow="password_change",
        userId=str(user.id),
        username=user.nome,
        createdAfter=created_after.isoformat(),
        deletedCount=len(models),
        activeVersion=active_version,
    )
    return ModelRollbackOutcome(deleted_count=len(models), active_version=active_version)


def activate_model_version(username: str, version: int) -> ModeloBiometrico:
    user = _resolve_user(username=username)
    if user is None:
        raise UserNotFoundError(f"Usuário '{username}' não encontrado.")

    target = ModeloBiometrico.objects.filter(
        usuario=user,
        versao=version,
        criado_em__gte=user.data_mudanca_senha,
    ).first()
    if target is None:
        raise ModelVersionNotFoundError(
            f"Versão v{version} não encontrada para o usuário '{user.nome}'."
        )

    load_model(target)

    if target.caminho_arquivo:
        active_file = Path(target.caminho_arquivo).parent / "active_version.json"
        active_file.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(
            active_file,
            json.dumps({"active_version": int(version)}, indent=2),
        )

    return target
