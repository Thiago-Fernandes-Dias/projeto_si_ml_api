import hashlib
import hmac
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import sklearn
from django.conf import settings
from django.contrib.auth.models import User
from django.db import transaction
from django.db.models import Max

from biometrics.ml.training import TrainingResult
from biometrics.models import TrainedModel, TrainingTrigger

logger = logging.getLogger(__name__)


class ModelIntegrityError(RuntimeError):
    """Raised when a model file on disk is missing or fails SHA-256 verification."""


def _compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_model(
    user: User,
    result: TrainingResult,
    trigger: TrainingTrigger | str = TrainingTrigger.ENROLLMENT,
) -> TrainedModel:
    models_root = Path(settings.BIOMETRICS["MODELS_DIR"])
    user_dir = models_root / str(user.id)
    user_dir.mkdir(parents=True, exist_ok=True)

    with transaction.atomic():
        max_version = (
            TrainedModel.objects.select_for_update()
            .filter(user=user)
            .aggregate(max_v=Max("version"))["max_v"]
            or 0
        )
        next_version = int(max_version) + 1
        model_path = user_dir / f"v{next_version}.joblib"

        joblib.dump(result.model, model_path)
        digest = _compute_sha256(model_path)

        TrainedModel.objects.filter(user=user, is_active=True).update(is_active=False)
        trained_model = TrainedModel.objects.create(
            user=user,
            version=next_version,
            file_path=str(model_path),
            sha256=digest,
            sklearn_version=sklearn.__version__,
            trigger=str(trigger),
            training_stats=result.stats,
            is_active=True,
        )

    return trained_model


@lru_cache(maxsize=128)
def _load_model_cached(
    model_id: int,
    expected_sha256: str,
    file_path_str: str,
    saved_sklearn_version: str,
) -> Any:
    file_path = Path(file_path_str)
    if not file_path.is_file():
        raise ModelIntegrityError(f"Arquivo de modelo #{model_id} não encontrado em '{file_path}'.")

    actual_sha256 = _compute_sha256(file_path)
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ModelIntegrityError(
            f"Falha na verificação de integridade SHA-256 para o modelo #{model_id}."
        )

    current_sklearn_version = sklearn.__version__
    if saved_sklearn_version != current_sklearn_version:
        logger.warning(
            "Modelo #%s foi salvo com scikit-learn %s, mas a versão instalada é %s.",
            model_id,
            saved_sklearn_version,
            current_sklearn_version,
        )

    return joblib.load(file_path)


def load_model(trained_model: TrainedModel) -> Any:
    return _load_model_cached(
        int(trained_model.id),
        str(trained_model.sha256),
        str(trained_model.file_path),
        str(trained_model.sklearn_version),
    )


def clear_model_cache() -> None:
    _load_model_cached.cache_clear()
