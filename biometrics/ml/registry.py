import hashlib
import hmac
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import sklearn
from django.conf import settings
from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from biometrics.ml.training import TrainingResult
from biometrics.models import ModeloBiometrico, TrainingTrigger, User

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
) -> ModeloBiometrico:
    models_root = Path(settings.BIOMETRICS["MODELS_DIR"])
    user_dir = models_root / str(user.id)
    user_dir.mkdir(parents=True, exist_ok=True)

    with transaction.atomic():
        max_version = (
            ModeloBiometrico.objects.select_for_update()
            .filter(usuario=user)
            .aggregate(max_v=Max("versao"))["max_v"]
            or 0
        )
        next_version = int(max_version) + 1
        staging_path = user_dir / f".v{next_version}.tmp.joblib"
        joblib.dump(result.model, staging_path)
        digest = _compute_sha256(staging_path)

        model_path = user_dir / f"v{next_version}_{digest}.joblib"
        staging_path.replace(model_path)

        meta_path = Path(f"{model_path}.meta.json")
        meta_payload = {
            "sha256": digest,
            "sklearn_version": sklearn.__version__,
            "trigger": str(trigger),
            "training_stats": result.stats,
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

        algorithm_name = str(
            result.stats.get("algorithm", "M2005_IDB")
        )[:50]

        trained_model = ModeloBiometrico.objects.create(
            usuario=user,
            versao=next_version,
            algoritmo=algorithm_name,
            caminho_arquivo=str(model_path),
            criado_em=timezone.now(),
        )
        active_file = user_dir / "active_version.json"
        active_file.write_text(
            json.dumps({"active_version": next_version}, indent=2),
            encoding="utf-8",
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
    if not expected_sha256 or not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ModelIntegrityError(
            f"Falha na verificação de integridade SHA-256 para o modelo #{model_id}."
        )

    current_sklearn_version = sklearn.__version__
    if saved_sklearn_version and saved_sklearn_version != current_sklearn_version:
        logger.warning(
            "Modelo #%s foi salvo com scikit-learn %s, mas a versão instalada é %s.",
            model_id,
            saved_sklearn_version,
            current_sklearn_version,
        )

    return joblib.load(file_path)


def load_model(trained_model: ModeloBiometrico) -> Any:
    return _load_model_cached(
        int(trained_model.id),
        str(trained_model.sha256),
        str(trained_model.caminho_arquivo),
        str(trained_model.sklearn_version),
    )


def clear_model_cache() -> None:
    _load_model_cached.cache_clear()
