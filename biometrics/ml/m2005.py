from collections import deque
from collections.abc import Mapping
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

IMPOSTOR_LABEL = -1
GENUINE_LABEL = 1


class DigraphsMetrics:
    average: float
    std_dev: float
    median: float

    def __init__(self, average: float, std_dev: float, median: float) -> None:
        self.average = float(average)
        self.std_dev = float(std_dev)
        self.median = float(median)


def _coerce_dataframe(
    x: pd.DataFrame | pd.Series | np.ndarray,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, pd.Series):
        return pd.DataFrame([x.copy()])
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    columns = (
        list(feature_names)
        if feature_names is not None and len(feature_names) == arr.shape[1]
        else [str(i) for i in range(arr.shape[1])]
    )
    return pd.DataFrame(arr, columns=columns)


def _coerce_series(
    x: pd.Series | np.ndarray | Mapping[str, float],
    feature_names: list[str] | None = None,
) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, Mapping):
        return pd.Series({str(k): float(v) for k, v in x.items()}, dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64).ravel()
    index = (
        list(feature_names)
        if feature_names is not None and len(feature_names) == len(arr)
        else [str(i) for i in range(len(arr))]
    )
    return pd.Series(arr, index=index, dtype=np.float64)


class M2005Alg(BaseEstimator):
    """M2005 Statistical Classifier (Magalhães et al., 2005; Pisani, SBSeg 2024).

    Classifies keystroke patterns based on per-digraph mean, median,
    and standard deviation boundaries. Supports sliding/growing gallery
    updates and user-specific dynamic threshold calibration ('min', 'min_non_outlier').
    """

    threshold: float
    _base_threshold: float
    inclusion_threshold: float | None
    window_strategy: str
    window_size: int | None
    threshold_method: str | None
    dynamic_threshold: bool
    _digraphs_metrics: dict[str, DigraphsMetrics]
    _training_data: pd.DataFrame | None
    _gallery: list[pd.Series]
    _effective_window_size: int
    feature_names_: list[str]
    classes_: np.ndarray

    def __init__(
        self,
        threshold: float = 0.7,
        inclusion_threshold: float | None = None,
        window_strategy: str = "sliding",
        window_size: int | None = 50,
        threshold_method: str | None = "min_non_outlier",
        dynamic_threshold: bool = True,
    ) -> None:
        self.threshold = float(threshold)
        self._base_threshold = float(threshold)
        self.inclusion_threshold = inclusion_threshold
        self.window_strategy = window_strategy
        self.window_size = window_size
        self.threshold_method = threshold_method
        self.dynamic_threshold = dynamic_threshold
        self._digraphs_metrics = {}
        self._training_data = None
        self._gallery = []
        self._effective_window_size = window_size if window_size is not None else 50
        self.feature_names_ = []
        self.classes_ = np.array([IMPOSTOR_LABEL, GENUINE_LABEL])

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "inclusion_threshold": self.inclusion_threshold,
            "window_strategy": self.window_strategy,
            "window_size": self.window_size,
            "threshold_method": self.threshold_method,
            "dynamic_threshold": self.dynamic_threshold,
        }

    def set_params(self, **params: object) -> Self:
        if "threshold" in params:
            self.threshold = float(params["threshold"])  # type: ignore[arg-type]
            self._base_threshold = float(params["threshold"])  # type: ignore[arg-type]
        self.inclusion_threshold = params.get("inclusion_threshold", self.inclusion_threshold)  # type: ignore[assignment]
        self.window_strategy = str(params.get("window_strategy", self.window_strategy))
        self.window_size = params.get("window_size", self.window_size)  # type: ignore[assignment]
        self.threshold_method = params.get("threshold_method", self.threshold_method)  # type: ignore[assignment]
        self.dynamic_threshold = bool(params.get("dynamic_threshold", self.dynamic_threshold))
        return self

    def _compute_metrics_from_dataframe(self, df: pd.DataFrame) -> None:
        self._digraphs_metrics = {}
        self.feature_names_ = [str(col) for col in df.columns]
        for digraph_name, values in df.items():
            average = float(values.mean())
            std_dev = 0.0 if len(values) <= 1 else float(values.std(ddof=1))
            median = float(values.median())
            self._digraphs_metrics[str(digraph_name)] = DigraphsMetrics(average, std_dev, median)
        self._training_data = df

    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: list[int] | np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> Self:
        df = _coerce_dataframe(x, feature_names=feature_names)
        if y is not None:
            y_arr = np.asarray(y)
            if len(y_arr) == len(df) and np.any(y_arr == GENUINE_LABEL):
                df = df.iloc[y_arr == GENUINE_LABEL].reset_index(drop=True)

        self._effective_window_size = (
            self.window_size if self.window_size is not None else max(len(df), 1)
        )
        rows = [row.copy() for _, row in df.iterrows()]
        if self.window_strategy == "sliding" and len(rows) > self._effective_window_size:
            rows = rows[-self._effective_window_size :]
        self._gallery = rows

        gallery_df = pd.DataFrame(self._gallery) if self._gallery else df
        self._compute_metrics_from_dataframe(gallery_df)
        self.classes_ = np.array([IMPOSTOR_LABEL, GENUINE_LABEL])
        if self.threshold_method is not None:
            self.calibrate_threshold(method=self.threshold_method)
        return self

    def calculate_score(self, x: pd.Series | np.ndarray | Mapping[str, float]) -> float:
        names = self.feature_names_ or list(self._digraphs_metrics.keys())
        series = _coerce_series(x, feature_names=names)
        hits: list[bool] = []
        for digraph_name, value in series.items():
            digraph_metrics = self._digraphs_metrics.get(str(digraph_name))
            if digraph_metrics is None:
                continue
            av = digraph_metrics.average if digraph_metrics.average != 0 else 1.0
            std = digraph_metrics.std_dev
            md = digraph_metrics.median
            b1 = min(av, md) * (0.95 - std / av)
            b2 = max(av, md) * (1.05 + std / av)
            low, high = min(b1, b2), max(b1, b2)
            res = bool(low <= float(value) <= high)
            hits.append(res)
        length: int = len(hits)
        if length == 0:
            return 0.0
        score_sum: float = 1.0 if hits[0] else 0.0
        for i in range(1, length):
            if hits[i]:
                score_sum += 1.5 if hits[i - 1] else 1.0
        return float(score_sum / ((length - 1) * 1.5 + 1.0))

    def predict_score(self, x: pd.Series | np.ndarray | Mapping[str, float]) -> float:
        return self.calculate_score(x)

    def calibrate_threshold(
        self,
        method: str = "min_non_outlier",
        outlier_std_factor: float = 2.5,
    ) -> float:
        """Computes user-specific threshold from gallery scores (Pisani, SBSeg 2024)."""
        if not self._gallery:
            return self.threshold
        scores = np.array([self.calculate_score(sample) for sample in self._gallery], dtype=float)
        if method == "min" or len(scores) <= 1:
            raw_threshold = float(np.min(scores))
            self.threshold = float(min(raw_threshold, self._base_threshold))
            return self.threshold

        # Pisani (2024) min-non-outlier: S_j^n = {s in S_j | s > mean(S_j) - 2.5 * std(S_j)}
        lower_cutoff = float(np.mean(scores) - outlier_std_factor * np.std(scores, ddof=1))
        filtered_scores = scores[scores > lower_cutoff]
        raw_threshold = (
            float(np.min(filtered_scores)) if len(filtered_scores) > 0 else float(np.min(scores))
        )
        self.threshold = float(min(raw_threshold, self._base_threshold))
        return self.threshold

    def should_include_sample(self, x: pd.Series | pd.DataFrame | np.ndarray) -> bool:
        effective_threshold = (
            self.inclusion_threshold if self.inclusion_threshold is not None else self.threshold
        )
        if isinstance(x, pd.Series) or (isinstance(x, np.ndarray) and x.ndim == 1):
            return self.calculate_score(x) >= effective_threshold
        df = _coerce_dataframe(x, feature_names=self.feature_names_)
        return all(self.calculate_score(row) >= effective_threshold for _, row in df.iterrows())

    def adapt(
        self,
        x: pd.Series | pd.DataFrame | np.ndarray,
        window: str | None = None,
        max_size: int | None = None,
    ) -> None:
        strategy = window if window is not None else self.window_strategy
        if strategy == "none":
            return

        if isinstance(x, pd.Series) or (isinstance(x, np.ndarray) and x.ndim == 1):
            new_rows = [_coerce_series(x, feature_names=self.feature_names_).copy()]
        else:
            df = _coerce_dataframe(x, feature_names=self.feature_names_)
            new_rows = [row.copy() for _, row in df.iterrows()]

        window_limit = max_size if max_size is not None else self._effective_window_size

        if strategy == "sliding":
            for row in new_rows:
                if len(self._gallery) >= window_limit:
                    self._gallery.pop(0)
                self._gallery.append(row)
        elif strategy == "growing":
            self._gallery.extend(new_rows)

        if self._gallery:
            self._compute_metrics_from_dataframe(pd.DataFrame(self._gallery))
            if self.dynamic_threshold and self.threshold_method is not None:
                self.calibrate_threshold(method=self.threshold_method)

    def predict_vec(self, x: pd.Series | np.ndarray | Mapping[str, float]) -> int:
        score = self.calculate_score(x)
        return GENUINE_LABEL if score >= self.threshold else IMPOSTOR_LABEL

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        df = _coerce_dataframe(x, feature_names=self.feature_names_)
        return np.array([self.predict_vec(row) for _, row in df.iterrows()])

    def predict_proba(self, x: pd.DataFrame | np.ndarray | pd.Series) -> np.ndarray:
        df = _coerce_dataframe(x, feature_names=self.feature_names_)
        scores = np.array([self.calculate_score(row) for _, row in df.iterrows()], dtype=np.float64)
        return np.column_stack([1.0 - scores, scores])


class _WelfordOnlineStats:
    """Welford's online algorithm for mean and sample standard deviation per digraph."""

    count: int
    mean: dict[str, float]
    squared_dev: dict[str, float]

    def __init__(self, df: pd.DataFrame) -> None:
        self.count = int(len(df))
        self.mean = {str(col): float(val) for col, val in df.mean().items()}
        self.squared_dev = {str(col): 0.0 for col in df.columns}
        for _, row in df.iterrows():
            for col in df.columns:
                col_key = str(col)
                delta = float(row[col]) - self.mean[col_key]
                self.squared_dev[col_key] += delta * delta

    def update(self, row: pd.Series) -> None:
        self.count += 1
        for col in row.index:
            col_key = str(col)
            val = float(row[col])
            delta = val - self.mean[col_key]
            self.mean[col_key] += delta / self.count
            self.squared_dev[col_key] += delta * (val - self.mean[col_key])

    def get_metrics(self) -> dict[str, DigraphsMetrics]:
        metrics: dict[str, DigraphsMetrics] = {}
        for col, mean_val in self.mean.items():
            variance = (
                max(self.squared_dev[col], 0.0) / (self.count - 1) if self.count > 1 else 0.0
            )
            std_val = float(np.sqrt(variance))
            metrics[col] = DigraphsMetrics(mean_val, std_val, mean_val)
        return metrics


class AdaptiveM2005Alg(BaseEstimator):
    """M2005 with Double Parallel (DB) and Improved Double Parallel (IDB) (Pisani et al., 2015).

    Maintains two M2005 submodels per user:
      1. Growing model (Welford incremental stats in O(1) memory for IDB; full gallery for DB).
      2. Sliding model (bounded FIFO gallery of size window_size).
    The verification score is the arithmetic average of both submodels.
    """

    threshold: float
    inclusion_threshold: float | None
    variant: str
    window_size: int | None
    feature_names_: list[str]
    classes_: np.ndarray
    _sliding_gallery: deque[pd.Series]
    _growing_gallery: list[pd.Series]
    _welford: _WelfordOnlineStats | None
    _sliding_estimator: M2005Alg
    _growing_estimator: M2005Alg

    def __init__(
        self,
        threshold: float = 0.7,
        variant: str = "idb",
        window_size: int | None = None,
        inclusion_threshold: float | None = None,
        threshold_method: str | None = None,
        dynamic_threshold: bool = False,
    ) -> None:
        normalized_variant = str(variant).lower()
        if normalized_variant not in {"idb", "db"}:
            raise ValueError(f"Variante inválida para AdaptiveM2005Alg: '{variant}'.")
        self.threshold = float(threshold)
        self._base_threshold = float(threshold)
        self.inclusion_threshold = (
            float(inclusion_threshold) if inclusion_threshold is not None else None
        )
        self.variant = normalized_variant
        self.window_size = int(window_size) if window_size is not None else None
        self.threshold_method = threshold_method
        self.dynamic_threshold = bool(dynamic_threshold)
        self.feature_names_ = []
        self.classes_ = np.array([IMPOSTOR_LABEL, GENUINE_LABEL])
        self._sliding_gallery = deque(maxlen=self.window_size)
        self._growing_gallery = []
        self._welford = None
        self._sliding_estimator = M2005Alg(
            threshold=self.threshold,
            window_strategy="none",
            window_size=None,
            threshold_method=None,
            dynamic_threshold=False,
        )
        self._growing_estimator = M2005Alg(
            threshold=self.threshold,
            window_strategy="none",
            window_size=None,
            threshold_method=None,
            dynamic_threshold=False,
        )

    @property
    def _gallery(self) -> list[pd.Series]:
        return list(self._sliding_gallery)

    @property
    def growing_count(self) -> int:
        if self.variant == "idb" and self._welford is not None:
            return int(self._welford.count)
        return len(self._growing_gallery)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "inclusion_threshold": self.inclusion_threshold,
            "variant": self.variant,
            "window_size": self.window_size,
            "threshold_method": self.threshold_method,
            "dynamic_threshold": self.dynamic_threshold,
        }

    def set_params(self, **params: object) -> Self:
        if "threshold" in params:
            self.threshold = float(params["threshold"])  # type: ignore[arg-type]
            self._base_threshold = self.threshold
            self._sliding_estimator.threshold = self.threshold
            self._growing_estimator.threshold = self.threshold
        if "inclusion_threshold" in params:
            val = params["inclusion_threshold"]
            self.inclusion_threshold = float(val) if val is not None else None  # type: ignore[arg-type]
        if "variant" in params:
            normalized = str(params["variant"]).lower()
            if normalized not in {"idb", "db"}:
                raise ValueError(f"Variante inválida para AdaptiveM2005Alg: '{normalized}'.")
            self.variant = normalized
        if "window_size" in params:
            ws = params["window_size"]
            self.set_window_size(int(ws) if ws is not None else None)  # type: ignore[arg-type]
        if "threshold_method" in params:
            self.threshold_method = params["threshold_method"]  # type: ignore[assignment]
        if "dynamic_threshold" in params:
            self.dynamic_threshold = bool(params["dynamic_threshold"])
        return self

    def set_window_size(self, window_size: int | None) -> None:
        self.window_size = int(window_size) if window_size is not None else None
        if self.window_size is not None:
            self._sliding_gallery = deque(self._sliding_gallery, maxlen=self.window_size)

    def _sync_growing_metrics(self) -> None:
        if self._welford is not None:
            self._growing_estimator._digraphs_metrics = self._welford.get_metrics()
            self._growing_estimator.feature_names_ = list(self.feature_names_)

    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: list[int] | np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> Self:
        df = _coerce_dataframe(x, feature_names=feature_names)
        if y is not None:
            y_arr = np.asarray(y)
            if len(y_arr) == len(df) and np.any(y_arr == GENUINE_LABEL):
                df = df.iloc[y_arr == GENUINE_LABEL].reset_index(drop=True)

        self.feature_names_ = [str(col) for col in df.columns]
        self.classes_ = np.array([IMPOSTOR_LABEL, GENUINE_LABEL])
        effective_maxlen = self.window_size if self.window_size is not None else max(len(df), 1)
        self._sliding_gallery = deque(
            [row.copy() for _, row in df.iterrows()],
            maxlen=effective_maxlen,
        )
        self._sliding_estimator.fit(pd.DataFrame(list(self._sliding_gallery)))

        if self.variant == "idb":
            self._growing_gallery = []
            self._welford = _WelfordOnlineStats(df)
            self._sync_growing_metrics()
        else:
            self._welford = None
            self._growing_gallery = [row.copy() for _, row in df.iterrows()]
            self._growing_estimator.fit(pd.DataFrame(self._growing_gallery))

        if self.threshold_method is not None:
            self.calibrate_threshold(method=self.threshold_method)

        return self

    def calculate_score(self, x: pd.Series | np.ndarray | Mapping[str, float]) -> float:
        names = self.feature_names_ or list(self._sliding_estimator.feature_names_)
        series = _coerce_series(x, feature_names=names)
        score_sliding = self._sliding_estimator.calculate_score(series)
        score_growing = self._growing_estimator.calculate_score(series)
        return float((score_sliding + score_growing) / 2.0)

    def predict_score(self, x: pd.Series | np.ndarray | Mapping[str, float]) -> float:
        return self.calculate_score(x)

    def calibrate_threshold(
        self,
        method: str = "min_non_outlier",
        outlier_std_factor: float = 2.5,
    ) -> float:
        """Calibrates the verification threshold from the genuine gallery scores."""
        if not self._sliding_gallery:
            return self.threshold

        scores = np.array(
            [self.calculate_score(sample) for sample in self._sliding_gallery],
            dtype=float,
        )

        if method == "min" or len(scores) <= 1:
            raw_threshold = float(np.min(scores))
        else:
            lower_cutoff = float(
                np.mean(scores) - outlier_std_factor * np.std(scores, ddof=1)
            )
            filtered_scores = scores[scores > lower_cutoff]
            raw_threshold = (
                float(np.min(filtered_scores))
                if len(filtered_scores) > 0
                else float(np.min(scores))
            )

        self.threshold = float(min(raw_threshold, self._base_threshold))
        return self.threshold

    def should_include_sample(self, x: pd.Series | pd.DataFrame | np.ndarray) -> bool:
        effective_threshold = (
            self.inclusion_threshold if self.inclusion_threshold is not None else self.threshold
        )
        if isinstance(x, pd.Series) or (isinstance(x, np.ndarray) and x.ndim == 1):
            return self.calculate_score(x) >= effective_threshold
        df = _coerce_dataframe(x, feature_names=self.feature_names_)
        return all(self.calculate_score(row) >= effective_threshold for _, row in df.iterrows())

    def adapt(self, x: pd.Series | pd.DataFrame | np.ndarray) -> None:
        if isinstance(x, pd.Series) or (isinstance(x, np.ndarray) and x.ndim == 1):
            rows = [_coerce_series(x, feature_names=self.feature_names_).copy()]
        else:
            df = _coerce_dataframe(x, feature_names=self.feature_names_)
            rows = [row.copy() for _, row in df.iterrows()]

        if not rows:
            return

        for row in rows:
            self._sliding_gallery.append(row.copy())
            if self.variant == "idb":
                if self._welford is None:
                    self._welford = _WelfordOnlineStats(pd.DataFrame([row]))
                else:
                    self._welford.update(row)
                self._sync_growing_metrics()
            else:
                self._growing_gallery.append(row.copy())

        if self.variant == "db" and self._growing_gallery:
            self._growing_estimator.fit(pd.DataFrame(self._growing_gallery))

        if self._sliding_gallery:
            self._sliding_estimator.fit(pd.DataFrame(list(self._sliding_gallery)))

        if self.dynamic_threshold and self.threshold_method is not None:
            self.calibrate_threshold(method=self.threshold_method)

    def predict_vec(self, x: pd.Series | np.ndarray | Mapping[str, float]) -> int:
        score = self.calculate_score(x)
        return GENUINE_LABEL if score >= self.threshold else IMPOSTOR_LABEL

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        df = _coerce_dataframe(x, feature_names=self.feature_names_)
        return np.array([self.predict_vec(row) for _, row in df.iterrows()])

    def predict_proba(self, x: pd.DataFrame | np.ndarray | pd.Series) -> np.ndarray:
        df = _coerce_dataframe(x, feature_names=self.feature_names_)
        scores = np.array([self.calculate_score(row) for _, row in df.iterrows()], dtype=np.float64)
        return np.column_stack([1.0 - scores, scores])


M2005 = M2005Alg
ImprovedStatisticalAlg = M2005Alg
M2005_Pisani2024_Sliding_Dynamic = M2005Alg
M2005_IDB = AdaptiveM2005Alg

