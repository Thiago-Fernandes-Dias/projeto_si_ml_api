import math
import re
from collections.abc import Mapping, Sequence

import numpy as np

FEATURE_KEY_PATTERN = re.compile(r"^(H|DD|UD)\.[A-Za-z0-9_.\-]+$")
MAX_INTERVAL_SECONDS = 5.0


class FeatureValidationError(ValueError):
    """Raised when keystroke feature payloads violate naming, numeric, or range constraints."""


def validate_sample(features: Mapping[str, float]) -> None:
    if not isinstance(features, Mapping) or len(features) == 0:
        raise FeatureValidationError("O dicionário de features não pode ser vazio.")

    for key, raw_value in features.items():
        if not isinstance(key, str) or not FEATURE_KEY_PATTERN.match(key):
            raise FeatureValidationError(
                f"Chave de feature inválida '{key}'. Esperado prefixo H., DD. ou UD."
            )

        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise FeatureValidationError(f"Valor não numérico para a feature '{key}'.")

        value = float(raw_value)
        if not math.isfinite(value):
            raise FeatureValidationError(f"Valor não finito para a feature '{key}'.")

        prefix = key.split(".", 1)[0]
        if prefix in ("H", "DD"):
            if not (0.0 < value <= MAX_INTERVAL_SECONDS):
                raise FeatureValidationError(
                    f"Feature '{key}' fora do intervalo permitido "
                    f"(0, {MAX_INTERVAL_SECONDS}] s: {value}."
                )
        elif prefix == "UD":
            if not (-MAX_INTERVAL_SECONDS <= value <= MAX_INTERVAL_SECONDS):
                raise FeatureValidationError(
                    f"Feature '{key}' fora do intervalo permitido "
                    f"[-{MAX_INTERVAL_SECONDS}, {MAX_INTERVAL_SECONDS}] s: {value}."
                )


def build_schema(samples: Sequence[Mapping[str, float]]) -> list[str]:
    if not samples:
        raise FeatureValidationError("A lista de amostras para construção do schema está vazia.")

    first_sample = samples[0]
    validate_sample(first_sample)
    canonical_keys = set(first_sample.keys())

    for idx, sample in enumerate(samples[1:], start=2):
        validate_sample(sample)
        sample_keys = set(sample.keys())
        if sample_keys != canonical_keys:
            missing = sorted(canonical_keys - sample_keys)
            extra = sorted(sample_keys - canonical_keys)
            raise FeatureValidationError(
                f"Amostra #{idx} possui schema inconsistente. Ausentes: {missing}; Extras: {extra}."
            )

    return sorted(canonical_keys)


def vectorize(features: Mapping[str, float], schema: Sequence[str]) -> np.ndarray:
    if not schema:
        raise FeatureValidationError("O schema do usuário está vazio.")

    validate_sample(features)

    expected_keys = set(schema)
    actual_keys = set(features.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise FeatureValidationError(
            f"Features incompatíveis com o schema do usuário. Ausentes: {missing}; Extras: {extra}."
        )

    return np.array([float(features[name]) for name in schema], dtype=np.float64)


def mean_hold_time_ms(features: Mapping[str, float]) -> float:
    hold_values = [
        float(val)
        for key, val in features.items()
        if (
            isinstance(key, str)
            and key.startswith("H.")
            and isinstance(val, int | float)
            and not isinstance(val, bool)
        )
    ]
    if not hold_values:
        return 0.0
    return float(np.mean(hold_values) * 1000.0)
