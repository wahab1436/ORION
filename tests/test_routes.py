"""
tests/test_routes.py
--------------------
Integration tests for all ORION Flask API endpoints.
"""

import io
import json

import numpy as np
import pandas as pd
import pytest

from app import create_app


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# CSV generators
# ---------------------------------------------------------------------------

def make_kpi_csv(n=60):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "kpi_name": "revenue",
        "kpi_value": 100 + rng.normal(0, 3, n),
        "target_value": 100.0,
        "notes": "",
    })
    return df.to_csv(index=False).encode("utf-8")


def make_segment_csv(n=60):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(7)
    rows = []
    for seg in ["North", "South", "East"]:
        rows.append(pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "segment_type": "region",
            "segment_name": seg,
            "segment_kpi_value": 33 + rng.normal(0, 2, n),
            "segment_change": rng.normal(0, 0.05, n),
            "weight": 1.0 / 3.0,
        }))
    return pd.concat(rows, ignore_index=True).to_csv(index=False).encode("utf-8")


def make_event_csv():
    df = pd.DataFrame({
        "date": ["2023-02-01"],
        "event_type": ["campaign"],
        "event_name": ["Q1 Launch"],
        "affected_segments": ["North"],
        "magnitude": [0.15],
        "notes": ["Paid media"],
    })
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Upload tests
# ---------------------------------------------------------------------------

class TestUpload:

    def test_upload_kpi_success(self, client):
        response = client.post(
            "/upload-data",
            data={"file": (io.BytesIO(make_kpi_csv()), "kpi.csv"), "table_type": "kpi"},
            content_type="multipart/form-data",
        )
        data = response.get_json()
        assert response.status_code == 200
        assert data["success"] is True
        assert data["data"]["rows"] == 60

    def test_upload_segment_success(self, client):
        response = client.post(
            "/upload-data",
            data={"file": (io.BytesIO(make_segment_csv()), "segment.csv"), "table_type": "segment"},
            content_type="multipart/form-data",
        )
        assert response.get_json()["success"] is True

    def test_upload_invalid_table_type(self, client):
        response = client.post(
            "/upload-data",
            data={"file": (io.BytesIO(b"a,b,c\n1,2,3"), "x.csv"), "table_type": "unknown"},
            content_type="multipart/form-data",
        )
        assert response.get_json()["success"] is False

    def test_upload_missing_file(self, client):
        response = client.post("/upload-data", data={"table_type": "kpi"})
        assert response.get_json()["success"] is False

    def test_upload_missing_columns(self, client):
        bad_csv = b"col_a,col_b\n1,2\n3,4\n"
        response = client.post(
            "/upload-data",
            data={"file": (io.BytesIO(bad_csv), "bad.csv"), "table_type": "kpi"},
            content_type="multipart/form-data",
        )
        assert response.get_json()["success"] is False


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

def upload_all(client):
    client.post(
        "/upload-data",
        data={"file": (io.BytesIO(make_kpi_csv()), "kpi.csv"), "table_type": "kpi"},
        content_type="multipart/form-data",
    )
    client.post(
        "/upload-data",
        data={"file": (io.BytesIO(make_segment_csv()), "segment.csv"), "table_type": "segment"},
        content_type="multipart/form-data",
    )
    client.post(
        "/upload-data",
        data={"file": (io.BytesIO(make_event_csv()), "event.csv"), "table_type": "event"},
        content_type="multipart/form-data",
    )


class TestAnomalyEndpoint:

    def test_detect_success(self, client):
        upload_all(client)
        response = client.post(
            "/detect-anomaly",
            json={"kpi_name": "revenue"},
        )
        data = response.get_json()
        assert data["success"] is True
        assert "anomaly_count" in data["data"]

    def test_detect_missing_kpi(self, client):
        upload_all(client)
        response = client.post("/detect-anomaly", json={"kpi_name": "nonexistent_kpi"})
        assert response.get_json()["success"] is False

    def test_detect_no_data(self, client):
        response = client.post("/detect-anomaly", json={"kpi_name": "revenue"})
        assert response.get_json()["success"] is False


class TestRootCauseEndpoint:

    def test_root_cause_success(self, client):
        upload_all(client)
        response = client.post("/root-cause", json={"kpi_name": "revenue"})
        data = response.get_json()
        assert data["success"] is True
        assert "ranked_causes" in data["data"]

    def test_root_cause_with_date(self, client):
        upload_all(client)
        response = client.post("/root-cause", json={"kpi_name": "revenue", "anomaly_date": "2023-02-01"})
        assert response.get_json()["success"] is True


class TestSimulationEndpoint:

    def test_simulate_success(self, client):
        upload_all(client)
        response = client.post("/simulate-impact", json={
            "kpi_name": "revenue",
            "segment_name": "North",
            "change_pct": 10,
        })
        data = response.get_json()
        assert data["success"] is True
        assert "p50" in data["data"]

    def test_simulate_missing_segment(self, client):
        upload_all(client)
        response = client.post("/simulate-impact", json={
            "kpi_name": "revenue",
            "segment_name": "Nowhere",
            "change_pct": 10,
        })
        assert response.get_json()["success"] is False

    def test_simulate_invalid_change_pct(self, client):
        upload_all(client)
        response = client.post("/simulate-impact", json={
            "kpi_name": "revenue",
            "segment_name": "North",
            "change_pct": "not_a_number",
        })
        assert response.get_json()["success"] is False


class TestHealthEndpoint:

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.get_json()["status"] == "ok"
