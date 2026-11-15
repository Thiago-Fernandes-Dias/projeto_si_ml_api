import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from django.conf import settings

from biometrics.logging_utils import log_structured

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Classification:
    decision: str
    genuine_probability: float
    inference_time_ms: float


def classify(
    model: Any,
    vector: np.ndarray,
    threshold: float | None = None,
) -> Classification:
    resolved_threshold = (
        float(threshold)
        if threshold is not None
        else float(getattr(model, "threshold", settings.BIOMETRICS["DECISION_THRESHOLD"]))
    )

    arr = np.asarray(vector, dtype=np.float64).ravel()

    start = time.perf_counter()
    if hasattr(model, "predict_score"):
        genuine_probability = float(model.predict_score(arr))
    else:
        probabilities = model.predict_proba(arr.reshape(1, -1))
        genuine_probability = float(probabilities[0, -1])
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    decision = "GENUINE" if genuine_probability >= resolved_threshold else "IMPOSTOR"
    log_structured(
        logger,
        logging.DEBUG,
        "ml.classify.completed",
        "Classificação biométrica executada.",
        decision=decision,
        genuineProbability=round(genuine_probability, 4),
        threshold=round(resolved_threshold, 4),
        inferenceTimeMs=round(elapsed_ms, 3),
        vectorLength=int(arr.size),
        modelType=type(model).__name__,
    )
    return Classification(
        decision=decision,
        genuine_probability=genuine_probability,
        inference_time_ms=elapsed_ms,
    )
