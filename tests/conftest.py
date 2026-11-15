import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from django.utils import timezone

from biometrics.ml.registry import clear_model_cache
from biometrics.models import User


def create_user_in_db(username: str) -> User:
    now = timezone.now()
    return User.objects.create(
        id=uuid.uuid4(),
        nome=username.strip().lower(),
        data_mudanca_senha=now,
        biometria_cadastrada_em=None,
        tentativas_falhas=0,
        bloqueado_ate=None,
        criado_em=now,
    )

CMU_SAMPLE_FEATURES: dict[str, float] = {
    "H.period": 0.1491,
    "DD.period.t": 0.3979,
    "UD.period.t": 0.2488,
    "H.t": 0.1069,
    "DD.t.i": 0.1674,
    "UD.t.i": 0.0605,
    "H.i": 0.1169,
    "DD.i.e": 0.2212,
    "UD.i.e": 0.1043,
    "H.e": 0.1417,
    "DD.e.five": 1.1885,
    "UD.e.five": 1.0468,
    "H.five": 0.1146,
    "DD.five.Shift.r": 1.6055,
    "UD.five.Shift.r": 1.4909,
    "H.Shift.r": 0.1067,
    "DD.Shift.r.o": 0.7590,
    "UD.Shift.r.o": 0.6523,
    "H.o": 0.1016,
    "DD.o.a": 0.2136,
    "UD.o.a": 0.1120,
    "H.a": 0.1349,
    "DD.a.n": 0.1484,
    "UD.a.n": 0.0135,
    "H.n": 0.0932,
    "DD.n.l": 0.3515,
    "UD.n.l": 0.2583,
    "H.l": 0.1338,
    "DD.l.Return": 0.3509,
    "UD.l.Return": 0.2171,
    "H.Return": 0.0742,
}


def make_genuine_feature_dicts(
    count: int = 5,
    seed: int = 42,
    base: dict[str, float] | None = None,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    template = base if base is not None else CMU_SAMPLE_FEATURES
    samples: list[dict[str, float]] = []
    for _ in range(count):
        perturbed: dict[str, float] = {}
        for key, val in template.items():
            factor = float(rng.uniform(0.985, 1.015))
            perturbed[key] = round(val * factor, 4)
        samples.append(perturbed)
    return samples


def make_section_5_3_payload(
    username: str = "thiago_dias",
    count: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    feature_dicts = make_genuine_feature_dicts(count=count, seed=seed)
    samples = [
        {
            "repetition": idx,
            "features": feat,
            "rawTimings": [
                {"key": ".", "pressTime": 1024.12, "releaseTime": 1173.22},
                {"key": "t", "pressTime": 1422.02, "releaseTime": 1528.92},
            ],
        }
        for idx, feat in enumerate(feature_dicts, start=1)
    ]
    return {
        "username": username,
        "passwordLength": 10,
        "samplesCount": count,
        "samples": samples,
    }


@pytest.fixture(autouse=True)
def isolated_models_dir(tmp_path: Path, settings: Any) -> Path:
    models_dir = tmp_path / "models_data"
    models_dir.mkdir(parents=True, exist_ok=True)
    settings.BIOMETRICS = {
        **settings.BIOMETRICS,
        "MODELS_DIR": models_dir,
    }
    clear_model_cache()
    yield models_dir
    clear_model_cache()
