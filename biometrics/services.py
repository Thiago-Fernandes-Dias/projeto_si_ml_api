from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from django.conf import settings
from django.contrib.auth.models import User
from django.db import IntegrityError, transaction

from biometrics.ml.features import (
    FeatureValidationError,
    build_schema,
    mean_hold_time_ms,
    vectorize,
)
from biometrics.ml.inference import classify
from biometrics.ml.registry import load_model, save_model
from biometrics.ml.training import train_user_model
from biometrics.models import (
    BiometricProfile,
    KeystrokeSample,
    SampleLabel,
    SampleSource,
    TrainedModel,
    TrainingTrigger,
)


class UserAlreadyExistsError(Exception):
    """Raised when attempting to register an existing username."""


class UserNotFoundError(Exception):
    """Raised when the target username does not exist."""


class AlreadyEnrolledError(Exception):
    """Raised when a user attempts enrollment more than once."""


class NotEnrolledError(Exception):
    """Raised when a user attempts biometric authentication before completing enrollment."""


class InvalidCredentialsError(Exception):
    """Raised when username does not exist or password does not match."""


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
    model_version: int
    sample_label: str
    message: str


def register_user(username: str, password: str) -> User:
    clean_username = (username or "").strip()
    if not clean_username:
        raise ValueError("O campo 'username' é obrigatório.")
    if not password:
        raise ValueError("O campo 'password' é obrigatório.")

    if User.objects.filter(username=clean_username).exists():
        raise UserAlreadyExistsError(f"O usuário '{clean_username}' já está cadastrado.")

    try:
        with transaction.atomic():
            return User.objects.create_user(username=clean_username, password=password)
    except IntegrityError as exc:
        raise UserAlreadyExistsError(f"O usuário '{clean_username}' já está cadastrado.") from exc


def enroll_user(
    username: str,
    samples: Sequence[Mapping[str, Any]],
    samples_count: int | None = None,
    rng: np.random.Generator | None = None,
) -> TrainedModel:
    clean_username = (username or "").strip()
    user = User.objects.filter(username=clean_username).first()
    if user is None:
        raise UserNotFoundError(f"Usuário '{clean_username}' não encontrado.")

    if BiometricProfile.objects.filter(user=user).exists():
        raise AlreadyEnrolledError(f"O usuário '{clean_username}' já realizou o enrollment.")

    expected_count = int(settings.BIOMETRICS["ENROLLMENT_SAMPLES"])
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

    extracted_features: list[Mapping[str, float]] = []
    extracted_raw_timings: list[Any] = []

    for idx, item in enumerate(samples, start=1):
        if not isinstance(item, Mapping):
            raise EnrollmentValidationError(f"Amostra #{idx} tem formato inválido.")
        if "features" in item:
            features_obj = item["features"]
            raw_timings_obj = item.get("rawTimings", item.get("raw_timings"))
        else:
            features_obj = item
            raw_timings_obj = None

        if not isinstance(features_obj, Mapping):
            raise EnrollmentValidationError(f"O campo 'features' da amostra #{idx} é inválido.")

        extracted_features.append(features_obj)
        extracted_raw_timings.append(raw_timings_obj)

    try:
        schema = build_schema(extracted_features)
        vectors = [vectorize(feat, schema) for feat in extracted_features]
    except FeatureValidationError as exc:
        raise EnrollmentValidationError(str(exc)) from exc

    genuine_matrix = np.vstack(vectors)
    training_result = train_user_model(
        genuine=genuine_matrix,
        real_impostors=None,
        schema=schema,
        rng=rng,
    )

    with transaction.atomic():
        if BiometricProfile.objects.select_for_update().filter(user=user).exists():
            raise AlreadyEnrolledError(f"O usuário '{clean_username}' já realizou o enrollment.")

        BiometricProfile.objects.create(
            user=user,
            feature_names=schema,
        )
        for feat, raw_timing in zip(extracted_features, extracted_raw_timings, strict=True):
            KeystrokeSample.objects.create(
                user=user,
                features=dict(feat),
                raw_timings=raw_timing,
                source=SampleSource.ENROLLMENT,
                label=SampleLabel.GENUINE,
                genuine_probability=1.0,
                classified_by=None,
            )

        trained_model = save_model(
            user=user,
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
    username: str,
    password: str,
    features: Mapping[str, float],
    raw_timings: Any = None,
) -> AuthenticationOutcome:
    clean_username = (username or "").strip()
    user = User.objects.filter(username=clean_username).first()
    if user is None or not password or not user.check_password(password):
        raise InvalidCredentialsError("Credenciais textuais inválidas.")

    profile = BiometricProfile.objects.filter(user=user).first()
    active_model = (
        TrainedModel.objects.filter(user=user, is_active=True).order_by("-version").first()
    )
    if profile is None or active_model is None:
        raise NotEnrolledError(
            "Usuário não possui perfil biométrico ou modelo ativo. Conclua o enrollment primeiro."
        )

    vector = vectorize(features, profile.feature_names)
    estimator = load_model(active_model)
    threshold = float(getattr(estimator, "threshold", settings.BIOMETRICS["DECISION_THRESHOLD"]))
    classification = classify(estimator, vector, threshold=threshold)

    sample_label = label_for(classification.genuine_probability, threshold=threshold)
    KeystrokeSample.objects.create(
        user=user,
        features=dict(features),
        raw_timings=raw_timings,
        source=SampleSource.LOGIN,
        label=sample_label,
        genuine_probability=classification.genuine_probability,
        classified_by=active_model,
    )

    attempt_hold_ms = mean_hold_time_ms(features)
    enrollment_samples = list(
        KeystrokeSample.objects.filter(user=user, source=SampleSource.ENROLLMENT).values_list(
            "features", flat=True
        )[:10]
    )
    if enrollment_samples:
        user_hold_ms = float(np.mean([mean_hold_time_ms(f) for f in enrollment_samples]))
    else:
        user_hold_ms = attempt_hold_ms

    if classification.decision == "GENUINE":
        message = (
            f"Autenticação biométrica aprovada (score={classification.genuine_probability:.2%}, "
            f"modelo v{active_model.version})."
        )
    else:
        message = (
            f"Padrão de digitação incompatível com o perfil biométrico "
            f"(score={classification.genuine_probability:.2%}, limiar={threshold:.2%})."
        )

    return AuthenticationOutcome(
        decision=classification.decision,
        confidence=classification.genuine_probability,
        threshold=threshold,
        inference_time_ms=classification.inference_time_ms,
        mean_hold_time_ms=attempt_hold_ms,
        user_average_hold_time_ms=user_hold_ms,
        model_version=active_model.version,
        sample_label=str(sample_label),
        message=message,
    )


def retrain_user(
    user: User,
    trigger: TrainingTrigger | str = TrainingTrigger.SCHEDULED,
    force: bool = False,
    rng: np.random.Generator | None = None,
) -> TrainedModel | None:
    with transaction.atomic():
        profile = (
            BiometricProfile.objects.select_for_update(skip_locked=True).filter(user=user).first()
        )
        if profile is None:
            return None

        active_model = (
            TrainedModel.objects.filter(user=user, is_active=True).order_by("-version").first()
        )
        min_new_samples = int(settings.BIOMETRICS["RETRAIN_MIN_NEW_SAMPLES"])

        if not force and active_model is not None:
            new_labeled_count = KeystrokeSample.objects.filter(
                user=user,
                label__in=[SampleLabel.GENUINE, SampleLabel.IMPOSTOR],
                created_at__gt=active_model.created_at,
            ).count()
            if new_labeled_count < min_new_samples:
                return None

        genuine_window = int(settings.BIOMETRICS["RETRAIN_GENUINE_WINDOW"])
        impostor_window = int(settings.BIOMETRICS["RETRAIN_IMPOSTOR_WINDOW"])

        genuine_samples = list(
            KeystrokeSample.objects.filter(
                user=user,
                label=SampleLabel.GENUINE,
            ).order_by("-created_at", "-id")[:genuine_window]
        )
        if not genuine_samples:
            return None
        genuine_samples.reverse()

        impostor_samples = list(
            KeystrokeSample.objects.filter(
                user=user,
                label=SampleLabel.IMPOSTOR,
            ).order_by("-created_at", "-id")[:impostor_window]
        )
        impostor_samples.reverse()

        schema = list(profile.feature_names)
        genuine_matrix = np.vstack([vectorize(s.features, schema) for s in genuine_samples])
        if impostor_samples:
            impostor_matrix = np.vstack([vectorize(s.features, schema) for s in impostor_samples])
        else:
            impostor_matrix = np.empty((0, len(schema)), dtype=np.float64)

        training_result = train_user_model(
            genuine=genuine_matrix,
            real_impostors=impostor_matrix,
            schema=schema,
            rng=rng,
        )
        return save_model(user=user, result=training_result, trigger=trigger)


def retrain_due_users(now: Any = None, force: bool = False) -> list[TrainedModel]:
    retrained: list[TrainedModel] = []
    user_ids = list(BiometricProfile.objects.values_list("user_id", flat=True))
    for user in User.objects.filter(id__in=user_ids).order_by("id"):
        new_model = retrain_user(
            user=user,
            trigger=TrainingTrigger.SCHEDULED,
            force=force,
        )
        if new_model is not None:
            retrained.append(new_model)
    return retrained


def activate_model_version(username: str, version: int) -> TrainedModel:
    clean_username = (username or "").strip()
    user = User.objects.filter(username=clean_username).first()
    if user is None:
        raise UserNotFoundError(f"Usuário '{clean_username}' não encontrado.")

    target = TrainedModel.objects.filter(user=user, version=version).first()
    if target is None:
        raise ModelVersionNotFoundError(
            f"Versão v{version} não encontrada para o usuário '{clean_username}'."
        )

    load_model(target)

    with transaction.atomic():
        TrainedModel.objects.filter(user=user, is_active=True).update(is_active=False)
        target.is_active = True
        target.save(update_fields=["is_active"])

    return target
