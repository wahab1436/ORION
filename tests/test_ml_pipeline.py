"""
tests/test_ml_pipeline.py
-------------------------
Unit tests for AnomalyDetector, RootCauseAnalyser, and ImpactSimulator.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml_pipeline import AnomalyDetector, ImpactSimulator, RootCauseAnalyser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_series():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    values = 100 + np.random.default_rng(0).normal(0, 2, 60)
    return pd.Series(values, index=dates)


@pytest.fixture
def series_with_anomalies(clean_series):
    s = clean_series.copy()
    s.iloc[20] = 200.0
    s.iloc[45] = 5.0
    return s


@pytest.fixture
def kpi_df():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    return pd.DataFrame({
        "date": dates,
        "kpi_name": "revenue",
        "kpi_value": 100 + np.random.default_rng(1).normal(0, 3, 60),
        "target_value": 100.0,
    })


@pytest.fixture
def segment_df():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    rows = []
    rng = np.random.default_rng(2)
    for seg in ["North", "South", "East"]:
        rows.append(pd.DataFrame({
            "date": dates,
            "segment_type": "region",
            "segment_name": seg,
            "segment_kpi_value": 33 + rng.normal(0, 2, 60),
            "segment_change": rng.normal(0, 0.05, 60),
            "weight": 1.0 / 3.0,
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class TestAnomalyDetector:

    def test_returns_dataframe(self, clean_series):
        detector = AnomalyDetector()
        result = detector.detect(clean_series)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, clean_series):
        detector = AnomalyDetector()
        result = detector.detect(clean_series)
        for col in ("value", "is_anomaly", "anomaly_score", "zscore_flag", "stl_flag", "iforest_flag"):
            assert col in result.columns

    def test_detects_injected_anomalies(self, series_with_anomalies):
        detector = AnomalyDetector()
        result = detector.detect(series_with_anomalies)
        assert result["is_anomaly"].iloc[20] or result["is_anomaly"].iloc[45], \
            "At least one injected anomaly should be detected."

    def test_anomaly_score_range(self, clean_series):
        detector = AnomalyDetector()
        result = detector.detect(clean_series)
        assert result["anomaly_score"].between(0, 1).all()

    def test_too_short_series_raises(self):
        s = pd.Series([1.0, 2.0], index=pd.date_range("2023-01-01", periods=2, freq="D"))
        with pytest.raises(ValueError):
            AnomalyDetector().detect(s)


# ---------------------------------------------------------------------------
# RootCauseAnalyser
# ---------------------------------------------------------------------------

class TestRootCauseAnalyser:

    def test_returns_dict(self, kpi_df, segment_df):
        analyser = RootCauseAnalyser()
        result = analyser.analyse(kpi_df, segment_df)
        assert isinstance(result, dict)

    def test_ranked_causes_present(self, kpi_df, segment_df):
        analyser = RootCauseAnalyser()
        result = analyser.analyse(kpi_df, segment_df)
        assert "ranked_causes" in result

    def test_ranked_causes_ordered(self, kpi_df, segment_df):
        analyser = RootCauseAnalyser()
        result = analyser.analyse(kpi_df, segment_df)
        scores = [abs(c["contribution"]) for c in result["ranked_causes"]]
        assert scores == sorted(scores, reverse=True)

    def test_empty_segment_returns_gracefully(self, kpi_df):
        analyser = RootCauseAnalyser()
        result = analyser.analyse(kpi_df, pd.DataFrame())
        assert result["ranked_causes"] == []

    def test_anomaly_date_accepted(self, kpi_df, segment_df):
        analyser = RootCauseAnalyser()
        result = analyser.analyse(kpi_df, segment_df, anomaly_date="2023-02-01")
        assert "ranked_causes" in result


# ---------------------------------------------------------------------------
# ImpactSimulator
# ---------------------------------------------------------------------------

class TestImpactSimulator:

    def test_returns_expected_keys(self, kpi_df, segment_df):
        sim = ImpactSimulator()
        result = sim.simulate(kpi_df, segment_df, "North", 10.0)
        for key in ("current_kpi", "regression_estimate", "simulated_mean", "p10", "p50", "p90"):
            assert key in result

    def test_positive_change_increases_estimate(self, kpi_df, segment_df):
        sim = ImpactSimulator()
        up = sim.simulate(kpi_df, segment_df, "North", 50.0)
        down = sim.simulate(kpi_df, segment_df, "North", -50.0)
        assert up["regression_estimate"] > down["regression_estimate"]

    def test_unknown_segment_raises(self, kpi_df, segment_df):
        sim = ImpactSimulator()
        with pytest.raises(ValueError, match="not found"):
            sim.simulate(kpi_df, segment_df, "Nonexistent", 10.0)

    def test_percentile_ordering(self, kpi_df, segment_df):
        sim = ImpactSimulator()
        result = sim.simulate(kpi_df, segment_df, "North", 10.0)
        assert result["p10"] <= result["p50"] <= result["p90"]
