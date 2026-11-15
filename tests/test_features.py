import math

import numpy as np
import pytest

from biometrics.ml.features import (
    FeatureValidationError,
    mean_hold_time_ms,
    positional_schema,
    validate_vector,
)


def test_positional_schema_follows_cmu_3n_minus_2_order() -> None:
    assert positional_schema(4) == ["H.0", "DD.0.1", "UD.0.1", "H.1"]
    assert positional_schema(7) == [
        "H.0",
        "DD.0.1",
        "UD.0.1",
        "H.1",
        "DD.1.2",
        "UD.1.2",
        "H.2",
    ]


@pytest.mark.parametrize("length", [0, 1, 2, 3, 5, 6, 8])
def test_positional_schema_rejects_invalid_lengths(length: int) -> None:
    with pytest.raises(FeatureValidationError):
        positional_schema(length)


@pytest.mark.parametrize(
    "invalid_vector",
    [
        [],
        [0.1],
        [0.1, 0.2],
        [0.1, 0.2, 0.1],
        [0.1, 0.2, 0.1, 0.1, 0.2],
        [0.0, 0.2, 0.1, 0.1],
        [0.1, 0.0, -0.1, 0.1],
        [0.1, 0.2, -5.1, 0.1],
        [0.1, 5.1, 0.1, 0.1],
        [0.1, 0.2, 0.1, True],
        [0.1, 0.2, 0.1, "0.1"],
        [0.1, math.nan, 0.1, 0.1],
        [0.1, math.inf, 0.1, 0.1],
    ],
)
def test_validate_vector_enforces_3n_minus_2_structure_and_slot_bounds(
    invalid_vector: list[object],
) -> None:
    with pytest.raises(FeatureValidationError):
        validate_vector(invalid_vector)


def test_validate_vector_returns_numpy_float_vector() -> None:
    vector = validate_vector([0.12, 0.20, -0.02, 0.10])
    assert isinstance(vector, np.ndarray)
    assert vector.tolist() == [0.12, 0.20, -0.02, 0.10]


def test_mean_hold_time_ms_computes_average_over_hold_slots_only() -> None:
    vector = [0.100, 0.900, 0.800, 0.200]
    assert math.isclose(mean_hold_time_ms(vector), 150.0, rel_tol=1e-6)
