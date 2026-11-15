import math
from collections.abc import Sequence

import numpy as np

MAX_INTERVAL_SECONDS = 5.0


class FeatureValidationError(ValueError):
    """Raised when positional biometric vectors violate structure or range constraints."""


def positional_schema(length: int) -> list[str]:
    if length < 4 or (length + 2) % 3 != 0:
        raise FeatureValidationError(
            f"O vetor posicional deve seguir o formato CMU 3n-2 (mínimo 4; recebido {length})."
        )
    num_keys = (length + 2) // 3
    names: list[str] = []
    for idx in range(num_keys):
        names.append(f"H.{idx}")
        if idx < num_keys - 1:
            names.append(f"DD.{idx}.{idx + 1}")
            names.append(f"UD.{idx}.{idx + 1}")
    return names


def validate_vector(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    if isinstance(vector, str | bytes) or not isinstance(vector, Sequence | np.ndarray):
        raise FeatureValidationError("O vetor biométrico deve ser uma sequência numérica.")
    if len(vector) == 0:
        raise FeatureValidationError("O vetor biométrico não pode ser vazio.")
    if len(vector) < 4 or (len(vector) + 2) % 3 != 0:
        raise FeatureValidationError(
            f"O vetor biométrico deve seguir a estrutura posicional 3n-2 "
            f"(mínimo 4 elementos; recebido {len(vector)})."
        )

    values: list[float] = []
    for idx, raw_val in enumerate(vector):
        if isinstance(raw_val, bool) or not isinstance(raw_val, int | float | np.number):
            raise FeatureValidationError(f"Elemento não numérico na posição {idx} do vetor.")
        val = float(raw_val)
        if not math.isfinite(val):
            raise FeatureValidationError(f"Elemento não finito na posição {idx} do vetor.")
        slot = idx % 3
        if slot in (0, 1):
            if not (0.0 < val <= MAX_INTERVAL_SECONDS):
                raise FeatureValidationError(
                    f"Posição {idx} (H/DD) fora do intervalo (0, {MAX_INTERVAL_SECONDS}] s: {val}."
                )
        else:
            if not (-MAX_INTERVAL_SECONDS <= val <= MAX_INTERVAL_SECONDS):
                raise FeatureValidationError(
                    f"Posição {idx} (UD) fora do intervalo "
                    f"[-{MAX_INTERVAL_SECONDS}, {MAX_INTERVAL_SECONDS}] s: {val}."
                )
        values.append(val)

    return np.array(values, dtype=np.float64)


def mean_hold_time_ms(vector: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(vector, dtype=np.float64).ravel()
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr[0::3]) * 1000.0)
