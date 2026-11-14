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


M2005 = M2005Alg
ImprovedStatisticalAlg = M2005Alg
M2005_Pisani2024_Sliding_Dynamic = M2005Alg
