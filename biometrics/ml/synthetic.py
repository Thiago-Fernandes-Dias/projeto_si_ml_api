import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from django.conf import settings

from biometrics.ml.features import MAX_INTERVAL_SECONDS

MIN_POSITIVE_SECONDS = 1e-4
DEFAULT_MIN_SCALED_DISTANCE = 1.5


def scaled_manhattan_distance(
    candidate: np.ndarray,
    genuine_mean: np.ndarray,
    genuine_std: np.ndarray,
) -> float:
    scale = np.maximum(genuine_std, np.maximum(0.05 * np.abs(genuine_mean), 1e-3))
    return float(np.mean(np.abs(candidate - genuine_mean) / scale))


def load_population_priors(priors_path: Path | None = None) -> dict[str, dict[str, float]] | None:
    resolved_path = priors_path
    if resolved_path is None:
        biometrics_cfg = getattr(settings, "BIOMETRICS", {})
        resolved_path = biometrics_cfg.get("POPULATION_PRIORS_PATH")
    if resolved_path is None:
        return None
    path_obj = Path(resolved_path)
    if not path_obj.is_file():
        return None
    with path_obj.open("r", encoding="utf-8") as fp:
        data: dict[str, dict[str, float]] = json.load(fp)
    return data


def _sample_perturbation_candidate(
    genuine: np.ndarray,
    schema: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    base_idx = int(rng.integers(0, len(genuine)))
    base = genuine[base_idx].astype(np.float64, copy=True)

    # Permute timings within each feature group (H, DD, UD) so the synthetic
    # impostor exhibits a distinct relative keystroke rhythm (rank order) as well
    # as a shifted global tempo.
    for group_prefix in ("H", "DD", "UD"):
        group_indices = [i for i, k in enumerate(schema) if k.split(".", 1)[0] == group_prefix]
        if len(group_indices) > 1:
            shuffled_indices = rng.permutation(group_indices)
            base[group_indices] = base[shuffled_indices]

    if rng.random() < 0.5:
        tempo = float(rng.uniform(0.55, 0.80))
    else:
        tempo = float(rng.uniform(1.25, 1.90))

    noise = np.exp(rng.normal(0.0, 0.35, size=len(schema)))
    candidate = base * tempo * noise

    for idx, key in enumerate(schema):
        prefix = key.split(".", 1)[0]
        if prefix in ("H", "DD"):
            clipped = np.clip(abs(candidate[idx]), MIN_POSITIVE_SECONDS, MAX_INTERVAL_SECONDS)
            candidate[idx] = float(clipped)
        else:
            if abs(base[idx]) < 1e-3:
                candidate[idx] = float(rng.normal(0.08 * tempo, 0.04))
            clipped = np.clip(candidate[idx], -MAX_INTERVAL_SECONDS, MAX_INTERVAL_SECONDS)
            candidate[idx] = float(clipped)

    return candidate


def _sample_population_candidate(
    schema: Sequence[str],
    priors: dict[str, dict[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    candidate = np.empty(len(schema), dtype=np.float64)
    for idx, key in enumerate(schema):
        prefix = key.split(".", 1)[0]
        bucket = priors.get(prefix, {})
        if prefix in ("H", "DD"):
            mu = float(bucket.get("log_mean", -2.1))
            sigma = max(float(bucket.get("log_std", 0.45)), 0.05)
            val = float(rng.lognormal(mean=mu, sigma=sigma))
            candidate[idx] = float(np.clip(val, MIN_POSITIVE_SECONDS, MAX_INTERVAL_SECONDS))
        else:
            mean = float(bucket.get("mean", 0.18))
            std = max(float(bucket.get("std", 0.15)), 0.02)
            val = float(rng.normal(loc=mean, scale=std))
            candidate[idx] = float(np.clip(val, -MAX_INTERVAL_SECONDS, MAX_INTERVAL_SECONDS))
    return candidate


def generate_negatives(
    genuine: np.ndarray,
    schema: Sequence[str],
    count: int,
    rng: np.random.Generator | None = None,
    min_distance: float = DEFAULT_MIN_SCALED_DISTANCE,
    priors_path: Path | None = None,
) -> np.ndarray:
    num_features = len(schema)
    if count <= 0:
        return np.empty((0, num_features), dtype=np.float64)

    genuine_arr = np.asarray(genuine, dtype=np.float64)
    if genuine_arr.ndim != 2 or genuine_arr.shape[1] != num_features or len(genuine_arr) == 0:
        raise ValueError("Matriz de amostras genuínas inválida para geração de sintéticos.")

    generator = rng if rng is not None else np.random.default_rng(42)
    genuine_mean = genuine_arr.mean(axis=0)
    genuine_std = (
        genuine_arr.std(axis=0, ddof=1) if len(genuine_arr) > 1 else np.zeros(num_features)
    )

    priors = load_population_priors(priors_path=priors_path)
    population_quota = count // 2 if priors is not None else 0
    perturbation_quota = count - population_quota

    accepted: list[np.ndarray] = []
    quotas = ((perturbation_quota, "perturbation"), (population_quota, "population"))

    for target_quota, mode in quotas:
        generated_for_mode = 0
        attempts = 0
        max_attempts = max(target_quota * 50, 200)

        while generated_for_mode < target_quota and attempts < max_attempts:
            attempts += 1
            if mode == "population" and priors is not None:
                candidate = _sample_population_candidate(schema, priors, generator)
            else:
                candidate = _sample_perturbation_candidate(genuine_arr, schema, generator)

            if scaled_manhattan_distance(candidate, genuine_mean, genuine_std) >= min_distance:
                accepted.append(candidate)
                generated_for_mode += 1

        while generated_for_mode < target_quota:
            fallback = genuine_mean.copy()
            for idx, key in enumerate(schema):
                prefix = key.split(".", 1)[0]
                if prefix in ("H", "DD"):
                    shifted = max(fallback[idx] * 2.2, MIN_POSITIVE_SECONDS * 10)
                    clipped = np.clip(shifted, MIN_POSITIVE_SECONDS, MAX_INTERVAL_SECONDS)
                    fallback[idx] = float(clipped)
                else:
                    shifted = fallback[idx] * 2.2 + 0.25
                    clipped = np.clip(shifted, -MAX_INTERVAL_SECONDS, MAX_INTERVAL_SECONDS)
                    fallback[idx] = float(clipped)
            accepted.append(fallback)
            generated_for_mode += 1

    return np.vstack(accepted)
