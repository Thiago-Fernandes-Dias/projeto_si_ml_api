from typing import Any

import pytest
from django.core.management import call_command

from biometrics.models import KeystrokeSample, SampleLabel, SampleSource, TrainedModel
from biometrics.services import (
    activate_model_version,
    enroll_user,
    register_user,
    retrain_due_users,
)
from tests.conftest import CMU_SAMPLE_FEATURES, make_genuine_feature_dicts


def _create_enrolled_user(username: str = "retrain_user") -> Any:
    user = register_user(username=username, password="StrongPassword123")
    samples = make_genuine_feature_dicts(count=5, seed=101)
    enroll_user(username=username, samples=samples, samples_count=5)
    return user


@pytest.mark.django_db
def test_retrain_skips_users_without_enough_new_samples_and_supports_force() -> None:
    user = _create_enrolled_user("user_skip")

    retrained = retrain_due_users(force=False)
    assert retrained == []
    assert TrainedModel.objects.filter(user=user).count() == 1

    call_command("retrain_models", username="user_skip", force=True)
    assert TrainedModel.objects.filter(user=user).count() == 2
    active = TrainedModel.objects.get(user=user, is_active=True)
    assert active.version == 2


@pytest.mark.django_db
def test_retrain_respects_sample_windows_excludes_uncertain_and_supports_rollback(
    settings: Any,
) -> None:
    settings.BIOMETRICS = {
        **settings.BIOMETRICS,
        "RETRAIN_MIN_NEW_SAMPLES": 3,
        "RETRAIN_GENUINE_WINDOW": 6,
        "RETRAIN_IMPOSTOR_WINDOW": 2,
    }
    user = _create_enrolled_user("user_window")
    v1 = TrainedModel.objects.get(user=user, is_active=True)

    for _ in range(4):
        KeystrokeSample.objects.create(
            user=user,
            features=dict(CMU_SAMPLE_FEATURES),
            source=SampleSource.LOGIN,
            label=SampleLabel.UNCERTAIN,
            genuine_probability=0.68,
            classified_by=v1,
        )

    assert retrain_due_users(force=False) == []

    for factor in (1.00, 1.01, 0.99):
        KeystrokeSample.objects.create(
            user=user,
            features={k: round(v * factor, 4) for k, v in CMU_SAMPLE_FEATURES.items()},
            source=SampleSource.LOGIN,
            label=SampleLabel.GENUINE,
            genuine_probability=0.98,
            classified_by=v1,
        )

    for factor in (2.0, 2.2, 2.4):
        KeystrokeSample.objects.create(
            user=user,
            features={k: round(v * factor, 4) for k, v in CMU_SAMPLE_FEATURES.items()},
            source=SampleSource.LOGIN,
            label=SampleLabel.IMPOSTOR,
            genuine_probability=0.05,
            classified_by=v1,
        )

    retrained = retrain_due_users(force=False)
    assert len(retrained) == 1
    v2 = retrained[0]
    assert v2.version == 2
    assert v2.is_active is True
    assert v2.training_stats["algorithm"] == "M2005_Pisani2024_Sliding_Dynamic"
    assert v2.training_stats["genuine"] == 6
    assert v2.training_stats["real_impostors"] == 2

    call_command("activate_model", "user_window", 1)
    v1.refresh_from_db()
    v2.refresh_from_db()
    assert v1.is_active is True
    assert v2.is_active is False

    activated = activate_model_version("user_window", 2)
    assert activated.version == 2
    assert activated.is_active is True
