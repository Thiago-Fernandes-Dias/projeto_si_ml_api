from pathlib import Path
from typing import Any

import numpy as np
import pytest
from django.db import IntegrityError

from biometrics.ml.features import build_schema, vectorize
from biometrics.ml.registry import (
    ModelIntegrityError,
    _load_model_cached,
    clear_model_cache,
    load_model,
    save_model,
)
from biometrics.ml.training import train_user_model
from biometrics.models import TrainedModel, TrainingTrigger
from biometrics.services import get_active_model
from tests.conftest import create_user_in_db, make_genuine_feature_dicts


def _build_training_result() -> Any:
    dicts = make_genuine_feature_dicts(count=5, seed=7)
    schema = build_schema(dicts)
    genuine = np.vstack([vectorize(d, schema) for d in dicts])
    return train_user_model(genuine=genuine, schema=schema)


@pytest.mark.django_db
def test_save_model_increments_version_and_keeps_single_active() -> None:
    user = create_user_in_db(username="alice")
    result = _build_training_result()

    m1 = save_model(user, result, trigger=TrainingTrigger.ENROLLMENT)
    assert m1.version == 1
    assert m1.is_active is True

    m2 = save_model(user, result, trigger=TrainingTrigger.SCHEDULED)
    m1.refresh_from_db()
    assert m2.version == 2
    assert m2.is_active is True
    assert m1.is_active is False
    active_model = get_active_model(user)
    assert active_model is not None
    assert active_model.id == m2.id


@pytest.mark.django_db
def test_unique_version_constraint_enforced_by_database() -> None:
    user = create_user_in_db(username="bob")
    result = _build_training_result()
    m1 = save_model(user, result, trigger=TrainingTrigger.ENROLLMENT)

    with pytest.raises(IntegrityError):
        TrainedModel.objects.create(
            usuario=user,
            versao=m1.versao,
            algoritmo=m1.algoritmo,
            caminho_arquivo=m1.caminho_arquivo,
        )


@pytest.mark.django_db
def test_load_model_rejects_tampered_file_and_uses_lru_cache(caplog: Any) -> None:
    user = create_user_in_db(username="carol")
    result = _build_training_result()
    saved = save_model(user, result, trigger=TrainingTrigger.ENROLLMENT)
    clear_model_cache()

    loaded_1 = load_model(saved)
    loaded_2 = load_model(saved)
    assert loaded_1 is loaded_2
    assert _load_model_cached.cache_info().hits == 1

    saved.sklearn_version = "0.0.0-mismatch"
    clear_model_cache()
    load_model(saved)
    assert "scikit-learn" in caplog.text

    clear_model_cache()
    model_path = Path(saved.file_path)
    model_path.write_bytes(model_path.read_bytes() + b"tampered")

    with pytest.raises(ModelIntegrityError, match="SHA-256"):
        load_model(saved)
