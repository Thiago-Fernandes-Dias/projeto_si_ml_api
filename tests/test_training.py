import numpy as np
import pandas as pd
import pytest

from biometrics.ml.features import positional_schema
from biometrics.ml.inference import classify
from biometrics.ml.m2005 import AdaptiveM2005Alg, M2005Alg
from biometrics.ml.training import adapt_user_model, train_user_model
from tests.conftest import make_genuine_vectors


def test_train_user_model_separates_genuine_from_2x_slower_impostor() -> None:
    vectors = make_genuine_vectors(count=5, seed=42)
    schema = positional_schema(len(vectors[0]))
    genuine = np.vstack(vectors)

    result = train_user_model(
        genuine=genuine,
        schema=schema,
    )

    model = result.model
    assert isinstance(model, AdaptiveM2005Alg)
    assert len(model.classes_) == 2
    assert result.stats["algorithm"] == "M2005_IDB"
    assert result.stats["variant"] == "idb"
    assert result.stats["window_size"] == 50
    assert result.stats["genuine"] == 5
    assert result.stats["growing_count"] == 5
    assert result.stats["gallery_size"] == 5
    assert result.stats["base_threshold"] == 0.7
    assert result.stats["threshold_method"] == "min_non_outlier"
    assert result.stats["dynamic_threshold"] is True
    assert model.threshold <= 0.7
    assert "synthetic" not in result.stats
    assert "real_impostors" not in result.stats

    genuine_probe = genuine[0]
    genuine_classification = classify(model, genuine_probe)
    assert genuine_classification.decision == "GENUINE"
    assert genuine_classification.genuine_probability >= 0.90
    assert genuine_classification.inference_time_ms > 0.0

    slow_impostor_probe = genuine[0] * 2.0
    impostor_classification = classify(model, slow_impostor_probe)
    assert impostor_classification.decision == "IMPOSTOR"
    assert impostor_classification.genuine_probability < 0.25


def test_m2005_consecutive_hit_weighting_and_sliding_dynamic_threshold() -> None:
    df = pd.DataFrame(
        [
            {"H.a": 0.10, "DD.a.b": 0.20, "UD.a.b": -0.05},
            {"H.a": 0.10, "DD.a.b": 0.20, "UD.a.b": -0.05},
            {"H.a": 0.10, "DD.a.b": 0.20, "UD.a.b": -0.05},
        ]
    )
    model = M2005Alg(
        threshold=0.7,
        window_strategy="sliding",
        window_size=3,
        threshold_method="min_non_outlier",
        dynamic_threshold=True,
    ).fit(df)

    all_hit = pd.Series({"H.a": 0.10, "DD.a.b": 0.20, "UD.a.b": -0.05})
    assert np.isclose(model.calculate_score(all_hit), 1.0)
    assert np.isclose(model.threshold, 0.7)

    first_two_hit = pd.Series({"H.a": 0.10, "DD.a.b": 0.20, "UD.a.b": -0.40})
    assert np.isclose(model.calculate_score(first_two_hit), (1.0 + 1.5) / 4.0)

    first_and_third_hit = pd.Series({"H.a": 0.10, "DD.a.b": 0.90, "UD.a.b": -0.05})
    assert np.isclose(model.calculate_score(first_and_third_hit), (1.0 + 1.0) / 4.0)

    model.adapt(first_two_hit)
    assert len(model._gallery) == 3
    assert model.threshold <= 0.7


def test_adaptive_m2005_idb_welford_grows_while_sliding_stays_bounded(settings: object) -> None:
    vectors = make_genuine_vectors(count=6, seed=99)
    schema = positional_schema(len(vectors[0]))
    all_vecs = np.vstack(vectors)

    settings.BIOMETRICS = {  # type: ignore[attr-defined]
        **settings.BIOMETRICS,  # type: ignore[attr-defined]
        "RETRAIN_GENUINE_WINDOW": 4,
        "M2005": {
            **settings.BIOMETRICS["M2005"],  # type: ignore[attr-defined]
            "window_size": 4,
        },
    }

    initial = train_user_model(genuine=all_vecs[:3], schema=schema)
    assert isinstance(initial.model, AdaptiveM2005Alg)
    assert initial.model.growing_count == 3
    assert len(initial.model._gallery) == 3

    adapted = adapt_user_model(initial.model, new_genuine=all_vecs[3:], schema=schema)
    assert isinstance(adapted.model, AdaptiveM2005Alg)
    assert initial.model.growing_count == 3
    assert adapted.model.growing_count == 6
    assert len(adapted.model._gallery) == 4
    assert adapted.stats["growing_count"] == 6
    assert adapted.stats["gallery_size"] == 4


def test_adaptive_m2005_idb_matches_reference_construction_and_score_average() -> None:
    vectors = make_genuine_vectors(count=5, seed=11)
    schema = positional_schema(len(vectors[0]))
    matrix = np.vstack(vectors)
    data = pd.DataFrame(matrix, columns=schema)

    model = AdaptiveM2005Alg(threshold=0.7, variant="idb", window_size=50).fit(data)

    assert model.threshold == 0.7
    assert model._sliding_estimator.window_strategy == "none"
    assert model._sliding_estimator.threshold_method is None
    assert model._sliding_estimator.dynamic_threshold is False
    assert model._growing_estimator.window_strategy == "none"
    assert model._growing_estimator.threshold_method is None
    assert model._growing_estimator.dynamic_threshold is False

    probe = data.iloc[0]
    expected = (
        model._sliding_estimator.calculate_score(probe)
        + model._growing_estimator.calculate_score(probe)
    ) / 2.0
    assert np.isclose(model.calculate_score(probe), expected)

    model.adapt(data.iloc[[1]])
    assert model.threshold == 0.7
    assert model.growing_count == 6


def _calibration_gallery() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    columns = [f"f{i}" for i in range(28)]
    base = rng.normal(0.2, 0.01, (9, 28))
    rows = list(base) + [base[0] * 1.2]
    return pd.DataFrame(rows, columns=columns).clip(lower=1e-3)


def test_adaptive_m2005_idb_calibrates_threshold_with_min_and_min_non_outlier() -> None:
    data = _calibration_gallery()
    reference = AdaptiveM2005Alg(0.7, "idb", 50).fit(data)
    scores = np.array([reference.calculate_score(row) for _, row in data.iterrows()])

    min_model = AdaptiveM2005Alg(
        threshold=0.7,
        variant="idb",
        window_size=50,
        threshold_method="min",
        dynamic_threshold=False,
    ).fit(data)
    assert min_model.threshold == pytest.approx(min(float(np.min(scores)), 0.7))

    lower_cutoff = float(np.mean(scores) - 2.5 * np.std(scores, ddof=1))
    filtered = scores[scores > lower_cutoff]
    expected_non_outlier = min(float(np.min(filtered)), 0.7)
    non_outlier_model = AdaptiveM2005Alg(
        threshold=0.7,
        variant="idb",
        window_size=50,
        threshold_method="min_non_outlier",
        dynamic_threshold=False,
    ).fit(data)
    assert non_outlier_model.threshold == pytest.approx(expected_non_outlier)
    assert min_model.threshold < non_outlier_model.threshold


def test_adaptive_m2005_idb_dynamic_threshold_recalibrates_after_adaptation() -> None:
    data = _calibration_gallery()
    static_model = AdaptiveM2005Alg(
        threshold=0.7,
        variant="idb",
        window_size=50,
        threshold_method="min",
        dynamic_threshold=False,
    ).fit(data)
    static_threshold = static_model.threshold
    static_model.adapt(data.iloc[[0]])
    assert static_model.threshold == static_threshold

    dynamic_model = AdaptiveM2005Alg(
        threshold=0.7,
        variant="idb",
        window_size=50,
        threshold_method="min",
        dynamic_threshold=True,
    ).fit(data)
    dynamic_model.adapt(data.iloc[[0]])
    scores = np.array(
        [dynamic_model.calculate_score(sample) for sample in dynamic_model._gallery]
    )
    assert dynamic_model.threshold == pytest.approx(min(float(np.min(scores)), 0.7))
