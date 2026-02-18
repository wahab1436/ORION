"""
ml_pipeline.py
--------------
Core analytical pipeline for ORION.

Modules
-------
- AnomalyDetector   : Z-score, STL decomposition, Isolation Forest
- RootCauseAnalyser : Gradient Boosting + SHAP + Bayesian segment weighting
- ImpactSimulator   : Monte Carlo projection and regression-based scenario analysis
- ModelStore        : Versioned persistence of trained models via joblib
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

class ModelStore:
    """Persist and load versioned ML models using joblib."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def save(self, model: Any, name: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        filename = f"{name}_{timestamp}.joblib"
        path = os.path.join(self.model_dir, filename)
        joblib.dump(model, path)
        logger.info("Model saved: %s", path)
        return path

    def load_latest(self, name: str) -> Optional[Any]:
        candidates = sorted(
            [f for f in os.listdir(self.model_dir) if f.startswith(name) and f.endswith(".joblib")],
            reverse=True,
        )
        if not candidates:
            return None
        path = os.path.join(self.model_dir, candidates[0])
        logger.info("Loading model: %s", path)
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _engineer_features(series: pd.Series, windows: List[int] = (3, 7, 14)) -> pd.DataFrame:
    """
    Derive rolling statistics and lag features from a univariate time series.
    Index must be a DatetimeIndex.
    """
    df = pd.DataFrame({"value": series})
    df["pct_change"] = df["value"].pct_change()
    for w in windows:
        df[f"rolling_mean_{w}"] = df["value"].rolling(w, min_periods=1).mean()
        df[f"rolling_std_{w}"] = df["value"].rolling(w, min_periods=1).std().fillna(0)
        df[f"lag_{w}"] = df["value"].shift(w)
    df = df.fillna(method="bfill").fillna(0)
    return df


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Detect anomalies in a KPI time series using three complementary methods:

    1. Z-score       : Statistical threshold on rolling Z-scores.
    2. STL           : Residuals from seasonal-trend decomposition.
    3. IsolationForest: Ensemble anomaly detection on engineered features.

    A point is flagged as anomalous when at least two of the three methods
    agree (majority vote).
    """

    ZSCORE_THRESHOLD = 2.5
    STL_RESIDUAL_MULTIPLIER = 2.0
    IF_CONTAMINATION = 0.05

    def __init__(self, model_store: Optional[ModelStore] = None):
        self.model_store = model_store
        self._iso_forest: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None

    def fit(self, series: pd.Series) -> "AnomalyDetector":
        """Train the Isolation Forest component on the provided series."""
        features = _engineer_features(series).values
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(features)
        self._iso_forest = IsolationForest(
            n_estimators=200,
            contamination=self.IF_CONTAMINATION,
            random_state=42,
        )
        self._iso_forest.fit(scaled)
        if self.model_store:
            self.model_store.save({"iso_forest": self._iso_forest, "scaler": self._scaler}, "anomaly_detector")
        return self

    def detect(self, series: pd.Series) -> pd.DataFrame:
        """
        Run all three detection methods and return a DataFrame with per-method
        flags and a consensus 'is_anomaly' column.
        """
        if series.empty or len(series) < 3:
            raise ValueError("Series must contain at least 3 data points.")

        result = pd.DataFrame(index=series.index)
        result["value"] = series.values

        result["zscore_flag"] = self._zscore_method(series)
        result["stl_flag"] = self._stl_method(series)
        result["iforest_flag"] = self._isolation_forest_method(series)

        vote = result[["zscore_flag", "stl_flag", "iforest_flag"]].sum(axis=1)
        result["is_anomaly"] = vote >= 2
        result["anomaly_score"] = vote / 3.0

        logger.info(
            "Anomaly detection complete. Flagged %d / %d points.",
            result["is_anomaly"].sum(),
            len(result),
        )
        return result

    # --- Private detection methods ---

    def _zscore_method(self, series: pd.Series) -> pd.Series:
        rolling_mean = series.rolling(window=14, min_periods=3).mean()
        rolling_std = series.rolling(window=14, min_periods=3).std().replace(0, np.nan)
        z = ((series - rolling_mean) / rolling_std).fillna(0)
        return (z.abs() > self.ZSCORE_THRESHOLD).astype(int)

    def _stl_method(self, series: pd.Series) -> pd.Series:
        try:
            from statsmodels.tsa.seasonal import STL
            period = min(7, max(2, len(series) // 3))
            stl = STL(series, period=period, robust=True)
            result = stl.fit()
            residuals = result.resid
            threshold = self.STL_RESIDUAL_MULTIPLIER * residuals.std()
            return (residuals.abs() > threshold).astype(int)
        except Exception as exc:
            logger.warning("STL decomposition failed: %s. Falling back to zero flags.", exc)
            return pd.Series(0, index=series.index)

    def _isolation_forest_method(self, series: pd.Series) -> pd.Series:
        if self._iso_forest is None or self._scaler is None:
            self.fit(series)
        features = _engineer_features(series).values
        scaled = self._scaler.transform(features)
        predictions = self._iso_forest.predict(scaled)
        # IsolationForest returns -1 for anomalies, 1 for inliers
        return pd.Series((predictions == -1).astype(int), index=series.index)


# ---------------------------------------------------------------------------
# Root cause analysis
# ---------------------------------------------------------------------------

class RootCauseAnalyser:
    """
    Attribute root causes of an anomaly to segment contributions.

    Method
    ------
    1. Build a feature matrix from segment KPI values.
    2. Train a Gradient Boosting regressor against the KPI target.
    3. Compute SHAP values to measure per-segment contribution.
    4. Apply Bayesian weighting by segment weight.
    """

    def __init__(self, model_store: Optional[ModelStore] = None):
        self.model_store = model_store
        self._model: Optional[GradientBoostingRegressor] = None

    def analyse(
        self,
        kpi_df: pd.DataFrame,
        segment_df: pd.DataFrame,
        anomaly_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return ranked root causes with contribution scores.

        Parameters
        ----------
        kpi_df      : KPI table filtered to the relevant KPI.
        segment_df  : Segment table for that KPI.
        anomaly_date: ISO date string of the anomaly to explain (optional).
        """
        if segment_df.empty:
            return {"ranked_causes": [], "shap_values": {}, "notes": "No segment data available."}

        pivot = (
            segment_df.pivot_table(
                index="date", columns="segment_name", values="segment_kpi_value", aggfunc="mean"
            )
            .fillna(method="ffill")
            .fillna(0)
        )

        kpi_aligned = kpi_df.set_index("date")["kpi_value"].reindex(pivot.index).fillna(method="ffill").fillna(0)

        if len(pivot) < 5:
            return {"ranked_causes": [], "shap_values": {}, "notes": "Insufficient data for model training."}

        X = pivot.values
        y = kpi_aligned.values

        model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
        model.fit(X, y)
        self._model = model

        if self.model_store:
            self.model_store.save(model, "root_cause_gbm")

        shap_values = self._compute_shap(X, pivot.columns.tolist())

        if anomaly_date:
            target_date = pd.Timestamp(anomaly_date)
            if target_date in pivot.index:
                row_idx = pivot.index.get_loc(target_date)
                point_shap = {col: float(shap_values[row_idx, i]) for i, col in enumerate(pivot.columns)}
            else:
                point_shap = dict(zip(pivot.columns, shap_values.mean(axis=0).tolist()))
        else:
            point_shap = dict(zip(pivot.columns, shap_values.mean(axis=0).tolist()))

        # Apply Bayesian weighting from segment weights
        weights = (
            segment_df.groupby("segment_name")["weight"].mean().reindex(pivot.columns).fillna(1.0)
        )
        weighted_scores = {seg: point_shap[seg] * weights.get(seg, 1.0) for seg in point_shap}

        ranked = sorted(weighted_scores.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "ranked_causes": [{"segment": seg, "contribution": score} for seg, score in ranked],
            "shap_values": point_shap,
            "model_feature_importances": dict(zip(pivot.columns, model.feature_importances_.tolist())),
        }

    def _compute_shap(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            values = explainer.shap_values(X)
            return values
        except Exception as exc:
            logger.warning("SHAP computation failed: %s. Using feature importances as fallback.", exc)
            importances = self._model.feature_importances_
            return np.tile(importances, (X.shape[0], 1))


# ---------------------------------------------------------------------------
# Impact simulation
# ---------------------------------------------------------------------------

class ImpactSimulator:
    """
    Estimate the projected impact of operational changes on a KPI.

    Two simulation modes
    --------------------
    monte_carlo : Perturb the requested segment value within a confidence
                  interval and run N iterations to produce a distribution.
    regression  : Predict outcome via a lightweight linear projection.
    """

    N_SIMULATIONS = 2000

    def simulate(
        self,
        kpi_df: pd.DataFrame,
        segment_df: pd.DataFrame,
        segment_name: str,
        change_pct: float,
        n_simulations: int = N_SIMULATIONS,
    ) -> Dict[str, Any]:
        """
        Simulate the KPI impact of applying `change_pct` to `segment_name`.

        Returns mean, std, percentile distribution, and a regression estimate.
        """
        if kpi_df.empty or segment_df.empty:
            raise ValueError("KPI and segment data are required for simulation.")

        kpi_series = kpi_df.sort_values("date")["kpi_value"].values
        seg_series = (
            segment_df[segment_df["segment_name"] == segment_name]
            .sort_values("date")["segment_kpi_value"]
            .values
        )

        if len(seg_series) == 0:
            raise ValueError(f"Segment '{segment_name}' not found in segment data.")

        current_kpi = kpi_series[-1]
        current_seg = seg_series[-1]

        weight = (
            segment_df[segment_df["segment_name"] == segment_name]["weight"].mean()
        )
        if np.isnan(weight):
            weight = 1.0 / max(segment_df["segment_name"].nunique(), 1)

        volatility = np.std(kpi_series) if len(kpi_series) > 1 else current_kpi * 0.05

        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0, volatility, size=n_simulations)
        delta = current_seg * (change_pct / 100.0) * weight
        simulated_kpis = current_kpi + delta + noise

        regression_estimate = current_kpi + delta

        return {
            "segment_name": segment_name,
            "change_pct": change_pct,
            "current_kpi": float(current_kpi),
            "regression_estimate": float(regression_estimate),
            "simulated_mean": float(np.mean(simulated_kpis)),
            "simulated_std": float(np.std(simulated_kpis)),
            "p10": float(np.percentile(simulated_kpis, 10)),
            "p50": float(np.percentile(simulated_kpis, 50)),
            "p90": float(np.percentile(simulated_kpis, 90)),
            "n_simulations": n_simulations,
        }
