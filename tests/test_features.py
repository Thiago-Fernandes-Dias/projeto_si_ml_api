import math

import numpy as np
import pytest

from biometrics.ml.features import (
    FeatureValidationError,
    build_schema,
    mean_hold_time_ms,
    validate_sample,
    vectorize,
)
from tests.conftest import CMU_SAMPLE_FEATURES


def test_validate_sample_accepts_valid_features_including_negative_ud_and_indexed_suffix() -> None:
    sample = {
        "H.t.0": 0.1100,
        "H.t.1": 0.1050,
        "DD.t.e.0": 0.1800,
        "UD.t.e.0": -0.0250,
    }
    validate_sample(sample)


@pytest.mark.parametrize(
    "invalid_sample",
    [
        {},
        {"X.t": 0.1},
        {"H": 0.1},
        {"H.t": 0.0},
        {"H.t": -0.01},
        {"H.t": 5.01},
        {"DD.t.i": 0.0},
        {"DD.t.i": 6.0},
        {"UD.t.i": -5.01},
        {"UD.t.i": 5.01},
        {"H.t": math.nan},
        {"H.t": math.inf},
        {"H.t": True},
        {"H.t": "0.1"},  # type: ignore[dict-item]
    ],
)
def test_validate_sample_rejects_invalid_inputs(invalid_sample: dict[str, float]) -> None:
    with pytest.raises(FeatureValidationError):
        validate_sample(invalid_sample)


def test_build_schema_returns_sorted_canonical_order() -> None:
    s1 = {"UD.a.b.0": -0.01, "H.a.0": 0.12, "DD.a.b.0": 0.11}
    s2 = {"H.a.0": 0.13, "UD.a.b.0": 0.02, "DD.a.b.0": 0.15}
    schema = build_schema([s1, s2])
    assert schema == ["DD.a.b.0", "H.a.0", "UD.a.b.0"]


def test_build_schema_rejects_inconsistent_keys_across_samples() -> None:
    s1 = {"H.a.0": 0.12, "DD.a.b.0": 0.20, "UD.a.b.0": 0.08}
    s2 = {"H.a.0": 0.12, "DD.a.b.0": 0.20}
    with pytest.raises(FeatureValidationError, match="schema inconsistente"):
        build_schema([s1, s2])


def test_vectorize_produces_numpy_vector_and_rejects_missing_or_extra_keys() -> None:
    schema = build_schema([CMU_SAMPLE_FEATURES])
    vec = vectorize(CMU_SAMPLE_FEATURES, schema)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (len(CMU_SAMPLE_FEATURES),)
    assert np.isclose(vec[schema.index("H.period")], 0.1491)

    missing = dict(CMU_SAMPLE_FEATURES)
    missing.pop("H.period")
    with pytest.raises(FeatureValidationError, match="Ausentes"):
        vectorize(missing, schema)

    extra = {**CMU_SAMPLE_FEATURES, "H.extra.0": 0.12}
    with pytest.raises(FeatureValidationError, match="Extras"):
        vectorize(extra, schema)


def test_mean_hold_time_ms_computes_average_over_hold_features_only() -> None:
    features = {
        "H.a.0": 0.100,
        "H.b.0": 0.200,
        "DD.a.b.0": 0.900,
        "UD.a.b.0": 0.800,
    }
    assert math.isclose(mean_hold_time_ms(features), 150.0, rel_tol=1e-6)
