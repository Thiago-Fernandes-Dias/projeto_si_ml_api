from typing import Any

import pytest
from django.core.management import call_command

from biometrics.ml.features import features_to_positional_vector
from biometrics.ml.registry import load_model
from biometrics.models import Amostra, DecisaoBiometrica, OrigemAmostra, TrainedModel
from biometrics.services import (
    activate_model_version,
    enroll_user,
    get_active_model,
    retrain_due_users,
)
from tests.conftest import CMU_SAMPLE_FEATURES, create_user_in_db, make_genuine_feature_dicts


def _create_enrolled_user(username: str = "retrain_user") -> Any:
    user = create_user_in_db(username=username)
    samples = make_genuine_feature_dicts(count=5, seed=101)
    for rep, feat in enumerate(samples, start=1):
        Amostra.objects.create(
            usuario=user,
            origem=OrigemAmostra.CADASTRO,
            vetor=features_to_positional_vector(feat).tolist(),
            repeticao=rep,
        )
    enroll_user(username=username)
    return user


@pytest.mark.django_db
def test_retrain_skips_users_without_enough_new_samples_and_supports_force() -> None:
    user = _create_enrolled_user("user_skip")

    retrained = retrain_due_users(force=False)
    assert retrained == []
    assert TrainedModel.objects.filter(usuario=user).count() == 1

    call_command("retrain_models", username="user_skip", force=True)
    assert TrainedModel.objects.filter(usuario=user).count() == 2
    active = get_active_model(user)
    assert active is not None
    assert active.version == 2


@pytest.mark.django_db
def test_retrain_respects_genuine_window_excludes_uncertain_and_impostors_and_supports_rollback(
    settings: Any,
) -> None:
    settings.BIOMETRICS = {
        **settings.BIOMETRICS,
        "RETRAIN_MIN_NEW_SAMPLES": 3,
        "RETRAIN_GENUINE_WINDOW": 6,
    }
    user = _create_enrolled_user("user_window")
    v1 = get_active_model(user)
    assert v1 is not None

    base_vec = features_to_positional_vector(CMU_SAMPLE_FEATURES).tolist()

    for _ in range(4):
        Amostra.objects.create(
            usuario=user,
            origem=OrigemAmostra.LOGIN,
            vetor=base_vec,
            decisao=DecisaoBiometrica.GENUINE,
            confianca=0.68,
            modelo=v1,
        )

    for factor in (2.0, 2.2, 2.4):
        Amostra.objects.create(
            usuario=user,
            origem=OrigemAmostra.LOGIN,
            vetor=[v * factor for v in base_vec],
            decisao=DecisaoBiometrica.IMPOSTOR,
            confianca=0.05,
            modelo=v1,
        )

    assert retrain_due_users(force=False) == []

    for factor in (1.00, 1.01, 0.99):
        Amostra.objects.create(
            usuario=user,
            origem=OrigemAmostra.LOGIN,
            vetor=[v * factor for v in base_vec],
            decisao=DecisaoBiometrica.GENUINE,
            confianca=0.98,
            modelo=v1,
        )

    retrained = retrain_due_users(force=False)
    assert len(retrained) == 1
    v2 = retrained[0]
    assert v2.version == 2
    assert v2.is_active is True
    assert v2.training_stats["algorithm"] == "M2005_Pisani2024_Sliding_Dynamic"
    assert v2.training_stats["genuine"] == 6
    assert "real_impostors" not in v2.training_stats
    assert "synthetic" not in v2.training_stats

    retrained_estimator = load_model(v2)
    assert len(retrained_estimator._gallery) == 6
    impostor_score = retrained_estimator.calculate_score([v * 2.0 for v in base_vec])
    assert impostor_score < retrained_estimator.threshold

    call_command("activate_model", "user_window", 1)
    v1.refresh_from_db()
    v2.refresh_from_db()
    assert v1.is_active is True
    assert v2.is_active is False
    active = get_active_model(user)
    assert active is not None
    assert active.versao == 1

    activated = activate_model_version("user_window", 2)
    assert activated.version == 2
    assert activated.is_active is True
    active_now = get_active_model(user)
    assert active_now is not None
    assert active_now.versao == 2
