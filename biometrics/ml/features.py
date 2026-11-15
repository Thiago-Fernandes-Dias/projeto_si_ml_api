import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

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
    expected_order = list(first_sample.keys())
    canonical_keys = set(expected_order)

    for idx, sample in enumerate(samples[1:], start=2):
        validate_sample(sample)
        sample_order = list(sample.keys())
        sample_keys = set(sample_order)
        if sample_keys != canonical_keys:
            missing = sorted(canonical_keys - sample_keys)
            extra = sorted(sample_keys - canonical_keys)
            raise FeatureValidationError(
                f"Amostra #{idx} possui schema inconsistente. Ausentes: {missing}; Extras: {extra}."
            )
        if sample_order != expected_order:
            raise FeatureValidationError(
                f"Amostra #{idx} possui features fora da ordem esperada."
            )

    return expected_order


def vectorize(features: Mapping[str, float], schema: Sequence[str]) -> np.ndarray:
    if not schema:
        raise FeatureValidationError("O schema do usuário está vazio.")

    validate_sample(features)

    expected_order = list(schema)
    actual_order = list(features.keys())
    expected_keys = set(expected_order)
    actual_keys = set(actual_order)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise FeatureValidationError(
            f"Features incompatíveis com o schema do usuário. Ausentes: {missing}; Extras: {extra}."
        )
    if actual_order != expected_order:
        raise FeatureValidationError(
            "Features em ordem incompatível com o schema do usuário."
        )

    return np.array([float(features[name]) for name in expected_order], dtype=np.float64)


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
    if isinstance(vector, str | bytes | Mapping) or not isinstance(vector, Sequence | np.ndarray):
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


def extract_vector_from_raw_timings(raw_timings: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not isinstance(raw_timings, Sequence) or len(raw_timings) < 2 or len(raw_timings) > 128:
        raise FeatureValidationError("rawTimings deve conter entre 2 e 128 eventos de tecla.")

    parsed_events: list[tuple[float, float]] = []
    for idx, event in enumerate(raw_timings):
        if not isinstance(event, Mapping):
            raise FeatureValidationError(f"Evento #{idx + 1} de rawTimings inválido.")
        press_raw = event.get("pressTime", event.get("press_time"))
        release_raw = event.get("releaseTime", event.get("release_time", press_raw))
        if (
            press_raw is None
            or release_raw is None
            or isinstance(press_raw, bool)
            or isinstance(release_raw, bool)
            or not isinstance(press_raw, int | float)
            or not isinstance(release_raw, int | float)
        ):
            raise FeatureValidationError(f"Tempos inválidos no evento #{idx + 1} de rawTimings.")
        press = float(press_raw)
        release = float(release_raw)
        if not math.isfinite(press) or not math.isfinite(release) or release < press:
            raise FeatureValidationError(f"Cronologia inválida no evento #{idx + 1} de rawTimings.")
        parsed_events.append((press, release))

    parsed_events.sort(key=lambda item: item[0])
    num_keys = len(parsed_events)
    values = np.empty(num_keys * 3 - 2, dtype=np.float64)

    for idx, (press, release) in enumerate(parsed_events):
        pos = idx * 3
        hold_sec = max((release - press) / 1000.0, 1e-4)
        values[pos] = hold_sec
        if idx < num_keys - 1:
            next_press = parsed_events[idx + 1][0]
            dd_sec = max((next_press - press) / 1000.0, 1e-4)
            ud_sec = (next_press - release) / 1000.0
            values[pos + 1] = dd_sec
            values[pos + 2] = ud_sec

    return validate_vector(values)


def _validate_positional_key_order(keys: Sequence[str]) -> None:
    if len(keys) < 4 or (len(keys) + 2) % 3 != 0:
        raise FeatureValidationError(
            f"As features devem seguir o formato posicional 3n-2 (mínimo 4; recebido {len(keys)})."
        )
    expected_prefixes = ("H.", "DD.", "UD.")
    for idx, key in enumerate(keys):
        expected_prefix = expected_prefixes[idx % 3]
        if not key.startswith(expected_prefix):
            raise FeatureValidationError(
                f"Feature '{key}' na posição {idx} está fora da ordem posicional esperada "
                f"(esperado prefixo '{expected_prefix}')."
            )


def features_to_positional_vector(
    features: Mapping[str, float],
    schema: Sequence[str] | None = None,
) -> np.ndarray:
    if schema is not None:
        raw_vector = vectorize(features, schema)
        keys_source = list(schema)
    else:
        validate_sample(features)
        keys_source = list(features.keys())
        raw_vector = np.array([float(features[k]) for k in keys_source], dtype=np.float64)

    _validate_positional_key_order(keys_source)
    return validate_vector(raw_vector)


def mean_hold_time_ms(
    features_or_vector: Mapping[str, float] | Sequence[float] | np.ndarray,
) -> float:
    if isinstance(features_or_vector, Mapping):
        hold_values = [
            float(val)
            for key, val in features_or_vector.items()
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

    arr = np.asarray(features_or_vector, dtype=np.float64).ravel()
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr[0::3]) * 1000.0)
