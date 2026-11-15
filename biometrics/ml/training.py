import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from django.conf import settings

from biometrics.logging_utils import log_structured
from biometrics.ml.m2005 import AdaptiveM2005Alg, M2005Alg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    model: AdaptiveM2005Alg | M2005Alg
    stats: dict[str, Any]


@dataclass(frozen=True)
class M2005Config:
    algorithm: str
    variant: str
    threshold: float
    inclusion_threshold: float | None
    window_strategy: str
    window_size: int
    threshold_method: str | None
    dynamic_threshold: bool


def get_m2005_config() -> M2005Config:
    biometrics_cfg = getattr(settings, "BIOMETRICS", {})
    m2005_cfg = biometrics_cfg.get("M2005", {})
    raw_algorithm = str(m2005_cfg.get("algorithm", "M2005_IDB")).strip()
    variant = str(m2005_cfg.get("variant", "idb")).strip().lower()
    if raw_algorithm == "M2005_DB":
        variant = "db"
    elif raw_algorithm == "M2005_IDB":
        variant = "idb"

    raw_inclusion = m2005_cfg.get("inclusion_threshold")
    inclusion_threshold = float(raw_inclusion) if raw_inclusion is not None else None
    default_window = biometrics_cfg.get("RETRAIN_GENUINE_WINDOW", 50)
    raw_window_size = m2005_cfg.get("window_size")
    window_size = int(raw_window_size) if raw_window_size is not None else int(default_window)
    return M2005Config(
        algorithm=raw_algorithm,
        variant=variant,
        threshold=float(m2005_cfg.get("threshold", biometrics_cfg.get("DECISION_THRESHOLD", 0.7))),
        inclusion_threshold=inclusion_threshold,
        window_strategy=str(m2005_cfg.get("window_strategy", "sliding")),
        window_size=window_size,
        threshold_method=m2005_cfg.get("threshold_method", "min_non_outlier"),
        dynamic_threshold=bool(m2005_cfg.get("dynamic_threshold", True)),
    )


def _resolve_schema(num_features: int, schema: list[str] | None) -> list[str]:
    if schema is not None and len(schema) == num_features:
        return list(schema)
    return [f"H.f{i}" for i in range(num_features)]


def _build_training_stats(
    classifier: AdaptiveM2005Alg | M2005Alg,
    eval_df: pd.DataFrame,
    cfg: M2005Config,
) -> dict[str, Any]:
    effective_threshold = float(classifier.threshold)
    if not eval_df.empty:
        scores = np.array(
            [classifier.calculate_score(row) for _, row in eval_df.iterrows()],
            dtype=float,
        )
        mean_score = float(np.mean(scores))
        min_score = float(np.min(scores))
        acceptance_rate = float(np.mean(scores >= effective_threshold))
    else:
        mean_score = 1.0
        min_score = 1.0
        acceptance_rate = 1.0

    if isinstance(classifier, AdaptiveM2005Alg):
        alg_label = "M2005_IDB" if classifier.variant == "idb" else "M2005_DB"
        return {
            "algorithm": alg_label,
            "variant": classifier.variant,
            "window_size": int(classifier.window_size or len(classifier._gallery)),
            "inclusion_threshold": classifier.inclusion_threshold,
            "threshold": effective_threshold,
            "base_threshold": classifier._base_threshold,
            "threshold_method": classifier.threshold_method,
            "dynamic_threshold": classifier.dynamic_threshold,
            "gallery_size": int(len(classifier._gallery)),
            "growing_count": int(classifier.growing_count),
            "genuine": int(classifier.growing_count),
            "mean_genuine_score": mean_score,
            "min_genuine_score": min_score,
            "genuine_acceptance_rate": acceptance_rate,
        }

    return {
        "algorithm": "M2005_Pisani2024_Sliding_Dynamic",
        "window_strategy": cfg.window_strategy,
        "window_size": cfg.window_size,
        "threshold_method": cfg.threshold_method,
        "dynamic_threshold": cfg.dynamic_threshold,
        "threshold": effective_threshold,
        "gallery_size": int(len(classifier._gallery)),
        "genuine": int(len(eval_df)),
        "mean_genuine_score": mean_score,
        "min_genuine_score": min_score,
        "genuine_acceptance_rate": acceptance_rate,
    }


def train_user_model(
    genuine: np.ndarray,
    schema: list[str] | None = None,
) -> TrainingResult:
    genuine_arr = np.asarray(genuine, dtype=np.float64)
    if genuine_arr.ndim != 2 or len(genuine_arr) == 0:
        raise ValueError("São necessárias amostras genuínas 2D não vazias para treinar o modelo.")

    cfg = get_m2005_config()
    resolved_schema = _resolve_schema(genuine_arr.shape[1], schema)
    df_genuine = pd.DataFrame(genuine_arr, columns=resolved_schema)

    log_structured(
        logger,
        logging.DEBUG,
        "ml.train.started",
        "Treinando modelo biométrico do usuário.",
        algorithm=cfg.algorithm,
        variant=cfg.variant,
        sampleCount=int(genuine_arr.shape[0]),
        featureCount=int(genuine_arr.shape[1]),
        baseThreshold=cfg.threshold,
        windowSize=cfg.window_size,
    )

    if cfg.algorithm in {"M2005_IDB", "M2005_DB"}:
        classifier: AdaptiveM2005Alg | M2005Alg = AdaptiveM2005Alg(
            threshold=cfg.threshold,
            variant=cfg.variant,
            window_size=cfg.window_size,
            inclusion_threshold=cfg.inclusion_threshold,
            threshold_method=cfg.threshold_method,
            dynamic_threshold=cfg.dynamic_threshold,
        )
    else:
        threshold_method = (
            str(cfg.threshold_method) if cfg.threshold_method is not None else None
        )
        classifier = M2005Alg(
            threshold=cfg.threshold,
            window_strategy=cfg.window_strategy,
            window_size=cfg.window_size,
            threshold_method=threshold_method,
            dynamic_threshold=cfg.dynamic_threshold,
        )

    classifier.fit(df_genuine)
    stats = _build_training_stats(classifier, df_genuine, cfg)
    log_structured(
        logger,
        logging.DEBUG,
        "ml.train.completed",
        "Treinamento do modelo biométrico concluído.",
        algorithm=stats.get("algorithm"),
        threshold=round(float(stats.get("threshold", cfg.threshold)), 4),
        gallerySize=stats.get("gallery_size"),
        meanGenuineScore=round(float(stats.get("mean_genuine_score", 0.0)), 4),
        minGenuineScore=round(float(stats.get("min_genuine_score", 0.0)), 4),
        genuineAcceptanceRate=round(float(stats.get("genuine_acceptance_rate", 0.0)), 4),
    )
    return TrainingResult(model=classifier, stats=stats)


def adapt_user_model(
    estimator: AdaptiveM2005Alg,
    new_genuine: np.ndarray | None = None,
    schema: list[str] | None = None,
) -> TrainingResult:
    """Returns a deep-copied AdaptiveM2005Alg adapted incrementally with new genuine samples."""
    cfg = get_m2005_config()
    cloned = copy.deepcopy(estimator)
    cloned.set_window_size(cfg.window_size)
    new_samples_count = 0

    if new_genuine is not None and len(new_genuine) > 0:
        new_arr = np.asarray(new_genuine, dtype=np.float64)
        if new_arr.ndim != 2:
            raise ValueError("A matriz de novas amostras genuínas deve ser 2D.")
        new_samples_count = int(new_arr.shape[0])
        resolved_schema = (
            list(cloned.feature_names_)
            if len(cloned.feature_names_) == new_arr.shape[1]
            else _resolve_schema(new_arr.shape[1], schema)
        )
        df_new = pd.DataFrame(new_arr, columns=resolved_schema)
        cloned.adapt(df_new)

    eval_df = (
        pd.DataFrame(cloned._gallery)
        if cloned._gallery
        else pd.DataFrame(columns=cloned.feature_names_)
    )
    stats = _build_training_stats(cloned, eval_df, cfg)
    log_structured(
        logger,
        logging.DEBUG,
        "ml.adapt.completed",
        "Adaptação incremental do modelo biométrico concluída.",
        algorithm=stats.get("algorithm"),
        newSamplesCount=new_samples_count,
        gallerySize=stats.get("gallery_size"),
        growingCount=stats.get("growing_count"),
        threshold=round(float(stats.get("threshold", cfg.threshold)), 4),
    )
    return TrainingResult(model=cloned, stats=stats)
