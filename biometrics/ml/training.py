from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from django.conf import settings

from biometrics.ml.m2005 import M2005Alg
from biometrics.ml.synthetic import generate_negatives


@dataclass(frozen=True)
class TrainingResult:
    model: M2005Alg
    stats: dict[str, Any]


def train_user_model(
    genuine: np.ndarray,
    real_impostors: np.ndarray | None = None,
    schema: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> TrainingResult:
    genuine_arr = np.asarray(genuine, dtype=np.float64)
    if genuine_arr.ndim != 2 or len(genuine_arr) == 0:
        raise ValueError("São necessárias amostras genuínas 2D não vazias para treinar o modelo.")

    num_features = genuine_arr.shape[1]
    resolved_schema = (
        list(schema)
        if schema is not None and len(schema) == num_features
        else [f"H.f{i}" for i in range(num_features)]
    )

    if real_impostors is None or len(real_impostors) == 0:
        impostors_arr = np.empty((0, num_features), dtype=np.float64)
    else:
        impostors_arr = np.asarray(real_impostors, dtype=np.float64)

    generator = rng if rng is not None else np.random.default_rng(42)

    biometrics_cfg = getattr(settings, "BIOMETRICS", {})
    synthetic_min = int(biometrics_cfg.get("SYNTHETIC_NEGATIVES_MIN", 50))
    synthetic_ratio = int(biometrics_cfg.get("SYNTHETIC_NEGATIVES_RATIO", 5))
    synthetic_count = max(
        0,
        max(synthetic_min, synthetic_ratio * len(genuine_arr)) - len(impostors_arr),
    )

    synthetic_arr = generate_negatives(
        genuine=genuine_arr,
        schema=resolved_schema,
        count=synthetic_count,
        rng=generator,
    )
    negatives = (
        np.vstack([impostors_arr, synthetic_arr])
        if len(impostors_arr) > 0 and len(synthetic_arr) > 0
        else (impostors_arr if len(impostors_arr) > 0 else synthetic_arr)
    )

    m2005_cfg = biometrics_cfg.get("M2005", {})
    base_threshold = float(
        m2005_cfg.get("threshold", biometrics_cfg.get("DECISION_THRESHOLD", 0.7))
    )
    window_strategy = str(m2005_cfg.get("window_strategy", "sliding"))
    default_window = biometrics_cfg.get("RETRAIN_GENUINE_WINDOW", 50)
    window_size = int(m2005_cfg.get("window_size", default_window))
    threshold_method = m2005_cfg.get("threshold_method", "min_non_outlier")
    dynamic_threshold = bool(m2005_cfg.get("dynamic_threshold", True))

    df_genuine = pd.DataFrame(genuine_arr, columns=resolved_schema)
    classifier = M2005Alg(
        threshold=base_threshold,
        window_strategy=window_strategy,
        window_size=window_size,
        threshold_method=str(threshold_method) if threshold_method is not None else None,
        dynamic_threshold=dynamic_threshold,
    )
    classifier.fit(df_genuine)
    effective_threshold = float(classifier.threshold)

    genuine_scores = np.array(
        [classifier.calculate_score(row) for _, row in df_genuine.iterrows()],
        dtype=float,
    )
    tpr = float(np.mean(genuine_scores >= effective_threshold)) if len(genuine_scores) > 0 else 1.0

    if len(negatives) > 0:
        df_negatives = pd.DataFrame(negatives, columns=resolved_schema)
        impostor_scores = np.array(
            [classifier.calculate_score(row) for _, row in df_negatives.iterrows()],
            dtype=float,
        )
        tnr = float(np.mean(impostor_scores < effective_threshold))
        mean_impostor_score = float(np.mean(impostor_scores))
    else:
        tnr = 1.0
        mean_impostor_score = 0.0

    bacc = (tpr + tnr) / 2.0

    stats: dict[str, Any] = {
        "algorithm": "M2005_Pisani2024_Sliding_Dynamic",
        "window_strategy": window_strategy,
        "window_size": window_size,
        "threshold_method": threshold_method,
        "dynamic_threshold": dynamic_threshold,
        "threshold": effective_threshold,
        "gallery_size": int(len(classifier._gallery)),
        "genuine": int(len(genuine_arr)),
        "real_impostors": int(len(impostors_arr)),
        "synthetic": int(len(synthetic_arr)),
        "mean_genuine_score": float(np.mean(genuine_scores)),
        "mean_impostor_score": mean_impostor_score,
        "oob_score": float(bacc),
    }

    return TrainingResult(model=classifier, stats=stats)
