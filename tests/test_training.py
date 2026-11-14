import numpy as np
import pandas as pd

from biometrics.ml.features import build_schema, vectorize
from biometrics.ml.inference import classify
from biometrics.ml.m2005 import M2005Alg
from biometrics.ml.training import train_user_model
from tests.conftest import make_genuine_feature_dicts


def test_train_user_model_separates_genuine_from_2x_slower_impostor() -> None:
    dicts = make_genuine_feature_dicts(count=5, seed=42)
    schema = build_schema(dicts)
    genuine = np.vstack([vectorize(d, schema) for d in dicts])

    result = train_user_model(
        genuine=genuine,
        real_impostors=None,
        schema=schema,
        rng=np.random.default_rng(42),
    )

    model = result.model
    assert isinstance(model, M2005Alg)
    assert len(model.classes_) == 2
    assert result.stats["algorithm"] == "M2005_Pisani2024_Sliding_Dynamic"
    assert result.stats["window_strategy"] == "sliding"
    assert result.stats["threshold_method"] == "min_non_outlier"
    assert result.stats["dynamic_threshold"] is True
    assert result.stats["genuine"] == 5
    assert result.stats["synthetic"] >= 50

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
