import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from django.db import IntegrityError
from django.utils import timezone

from biometrics.ml.features import positional_schema
from biometrics.ml.registry import (
    ModelIntegrityError,
    _load_model_cached,
    clear_model_cache,
    load_model,
    save_model,
)
from biometrics.ml.training import train_user_model
from biometrics.models import ModeloBiometrico, TrainedModel, TrainingTrigger, User
from biometrics.services import (
    ModelVersionNotFoundError,
    activate_model_version,
    get_active_model,
    rollback_models_after,
)
from tests.conftest import create_user_in_db, make_genuine_vectors


def _build_training_result() -> Any:
    vectors = make_genuine_vectors(count=5, seed=7)
    schema = positional_schema(len(vectors[0]))
    genuine = np.vstack(vectors)
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


@pytest.mark.django_db
def test_save_model_rolls_back_artifacts_and_active_version_on_db_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = create_user_in_db(username="dave")
    result = _build_training_result()
    first = save_model(user, result, trigger=TrainingTrigger.ENROLLMENT)

    user_dir = Path(first.file_path).parent
    active_file = user_dir / "active_version.json"
    previous_active = active_file.read_text(encoding="utf-8")
    files_before = {path.name for path in user_dir.iterdir()}

    def fail_save(self: Any, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(TrainedModel, "save", fail_save)

    with pytest.raises(RuntimeError, match="database unavailable"):
        save_model(user, result, trigger=TrainingTrigger.SCHEDULED)

    assert {path.name for path in user_dir.iterdir()} == files_before
    assert active_file.read_text(encoding="utf-8") == previous_active
    assert TrainedModel.objects.filter(usuario=user).count() == 1


@pytest.mark.django_db
def test_get_active_model_ignores_models_created_before_password_change() -> None:
    user = create_user_in_db(username="eve")
    ModeloBiometrico.objects.create(
        usuario=user,
        versao=1,
        algoritmo="M2005_IDB",
        caminho_arquivo="old-model.joblib",
        criado_em=user.data_mudanca_senha - timedelta(days=1),
    )

    assert get_active_model(user) is None

    current = save_model(user, _build_training_result(), trigger=TrainingTrigger.ENROLLMENT)
    assert current.versao == 2
    active = get_active_model(user)
    assert active is not None
    assert active.id == current.id

    with pytest.raises(ModelVersionNotFoundError):
        activate_model_version("eve", 1)


@pytest.mark.django_db
def test_rollback_models_after_removes_new_versions_and_restores_active_version() -> None:
    user = create_user_in_db(username="frank")
    old_date = timezone.now() - timedelta(days=2)
    User.objects.filter(pk=user.pk).update(data_mudanca_senha=old_date)
    user.refresh_from_db()

    first = save_model(user, _build_training_result(), trigger=TrainingTrigger.ENROLLMENT)
    second = save_model(user, _build_training_result(), trigger=TrainingTrigger.ENROLLMENT)

    first_date = timezone.now() - timedelta(days=1)
    second_date = timezone.now() + timedelta(minutes=1)
    ModeloBiometrico.objects.filter(pk=first.pk).update(criado_em=first_date)
    ModeloBiometrico.objects.filter(pk=second.pk).update(criado_em=second_date)

    first_path = Path(first.file_path)
    second_path = Path(second.file_path)

    outcome = rollback_models_after(
        created_after=first_date + timedelta(hours=1),
        username="frank",
    )

    assert outcome.deleted_count == 1
    assert outcome.active_version == 1
    assert not second_path.exists()
    assert not Path(f"{second_path}.meta.json").exists()
    assert first_path.exists()

    active = get_active_model(user)
    assert active is not None
    assert active.id == first.id

    active_file = first_path.parent / "active_version.json"
    assert json.loads(active_file.read_text(encoding="utf-8"))["active_version"] == 1
