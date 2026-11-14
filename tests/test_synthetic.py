from pathlib import Path

import numpy as np
from django.core.management import call_command

from biometrics.ml.features import build_schema, vectorize
from biometrics.ml.synthetic import (
    DEFAULT_MIN_SCALED_DISTANCE,
    generate_negatives,
    scaled_manhattan_distance,
)
from tests.conftest import make_genuine_feature_dicts


def _prepare_genuine_matrix() -> tuple[np.ndarray, list[str]]:
    dicts = make_genuine_feature_dicts(count=5, seed=10)
    schema = build_schema(dicts)
    matrix = np.vstack([vectorize(d, schema) for d in dicts])
    return matrix, schema


def test_generate_negatives_shape_finiteness_bounds_and_min_distance() -> None:
    genuine, schema = _prepare_genuine_matrix()
    rng = np.random.default_rng(123)
    negatives = generate_negatives(genuine, schema, count=40, rng=rng)

    assert negatives.shape == (40, len(schema))
    assert np.all(np.isfinite(negatives))

    for col_idx, name in enumerate(schema):
        prefix = name.split(".", 1)[0]
        if prefix in ("H", "DD"):
            assert np.all(negatives[:, col_idx] > 0.0)
            assert np.all(negatives[:, col_idx] <= 5.0)
        else:
            assert np.all(negatives[:, col_idx] >= -5.0)
            assert np.all(negatives[:, col_idx] <= 5.0)

    mean = genuine.mean(axis=0)
    std = genuine.std(axis=0, ddof=1)
    for row in negatives:
        dist = scaled_manhattan_distance(row, mean, std)
        assert dist >= DEFAULT_MIN_SCALED_DISTANCE


def test_generate_negatives_reproducibility_with_seed() -> None:
    genuine, schema = _prepare_genuine_matrix()
    neg_a = generate_negatives(genuine, schema, count=20, rng=np.random.default_rng(999))
    neg_b = generate_negatives(genuine, schema, count=20, rng=np.random.default_rng(999))
    neg_c = generate_negatives(genuine, schema, count=20, rng=np.random.default_rng(1000))

    assert np.allclose(neg_a, neg_b)
    assert not np.allclose(neg_a, neg_c)


def test_generate_negatives_with_population_priors_command(tmp_path: Path) -> None:
    genuine, schema = _prepare_genuine_matrix()
    csv_path = tmp_path / "cmu_sample.csv"
    priors_path = tmp_path / "population_priors.json"

    header = ",".join(["subject", "sessionIndex", "rep", *schema])
    rows = [header]
    for rep, row in enumerate(genuine, start=1):
        vals = ",".join(f"{v:.4f}" for v in row)
        rows.append(f"s002,1,{rep},{vals}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    call_command("fit_population_priors", str(csv_path), output=str(priors_path))
    assert priors_path.is_file()

    negatives = generate_negatives(
        genuine,
        schema,
        count=30,
        rng=np.random.default_rng(77),
        priors_path=priors_path,
    )
    assert negatives.shape == (30, len(schema))
    assert np.all(np.isfinite(negatives))
