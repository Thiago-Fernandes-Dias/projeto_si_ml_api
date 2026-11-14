from pathlib import Path
from typing import Any

import numpy as np
import pytest
from django.contrib.auth.models import User
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
from tests.conftest import make_genuine_feature_dicts


def _build_training_result() -> Any:
    dicts = make_genuine_feature_dicts(count=5, seed=7)
    schema = build_schema(dicts)
    genuine = np.vstack([vectorize(d, schema) for d in dicts])
    return train_user_model(genuine=genuine, schema=schema, rng=np.random.default_rng(7))


@pytest.mark.django_db
def test_save_model_increments_version_and_keeps_single_active() -> None:
    user = User.objects.create_user(username="alice", password="secretPassword123")
    result = _build_training_result()

    m1 = save_model(user, result, trigger=TrainingTrigger.ENROLLMENT)
    assert m1.version == 1
    assert m1.is_active is True

    m2 = save_model(user, result, trigger=TrainingTrigger.SCHEDULED)
    m1.refresh_from_db()
    assert m2.version == 2
    assert m2.is_active is True
    assert m1.is_active is False
    assert TrainedModel.objects.filter(user=user, is_active=True).count() == 1


@pytest.mark.django_db
def test_single_active_model_constraint_enforced_by_database() -> None:
    user = User.objects.create_user(username="bob", password="secretPassword123")
    result = _build_training_result()
    m1 = save_model(user, result, trigger=TrainingTrigger.ENROLLMENT)
    save_model(user, result, trigger=TrainingTrigger.MANUAL)

    m1.is_active = True
    with pytest.raises(IntegrityError):
        m1.save()


@pytest.mark.django_db
def test_load_model_rejects_tampered_file_and_uses_lru_cache(caplog: Any) -> None:
    user = User.objects.create_user(username="carol", password="secretPassword123")
    result = _build_training_result()
    saved = save_model(user, result, trigger=TrainingTrigger.ENROLLMENT)
    clear_model_cache()

    loaded_1 = load_model(saved)
    loaded_2 = load_model(saved)
    assert loaded_1 is loaded_2
    assert _load_model_cached.cache_info().hits == 1

    saved.sklearn_version = "0.0.0-mismatch"
    saved.save(update_fields=["sklearn_version"])
    clear_model_cache()
    load_model(saved)
    assert "scikit-learn" in caplog.text

    clear_model_cache()
    model_path = Path(saved.file_path)
    model_path.write_bytes(model_path.read_bytes() + b"tampered")

    with pytest.raises(ModelIntegrityError, match="SHA-256"):
        load_model(saved)
