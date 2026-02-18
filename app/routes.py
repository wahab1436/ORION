"""
routes.py
---------
Flask Blueprint containing all ORION API endpoints.

Endpoints
---------
GET  /                  Dashboard view
POST /upload-data       Parse, validate, and cache CSV data
POST /detect-anomaly    Identify anomalies in a KPI series
POST /root-cause        Rank root causes for a given KPI / date
POST /simulate-impact   Monte Carlo + regression impact estimate
POST /recommendation    LLM-generated recommendation via Gemini API
GET  /health            Health check
"""

import io
import json
import logging
import os
import time
from typing import Dict

import pandas as pd
from flask import Blueprint, current_app, jsonify, render_template, request

from app.ml_pipeline import AnomalyDetector, ImpactSimulator, ModelStore, RootCauseAnalyser
from app.utils import (
    build_error_response,
    build_success_response,
    coerce_dataframe,
    sanitize_string,
    validate_dataframe,
)

logger = logging.getLogger(__name__)
bp = Blueprint("orion", __name__)

# ---------------------------------------------------------------------------
# In-memory session cache (replace with Redis or DB for production at scale)
# ---------------------------------------------------------------------------
_cache: Dict[str, pd.DataFrame] = {}


def _get_model_store() -> ModelStore:
    return ModelStore(model_dir=current_app.config["MODEL_DIR"])


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------

@bp.before_request
def _record_start():
    request._start_time = time.perf_counter()


@bp.after_request
def _log_request(response):
    elapsed = time.perf_counter() - getattr(request, "_start_time", time.perf_counter())
    logger.info(
        "%s %s %d %.4fs",
        request.method,
        request.path,
        response.status_code,
        elapsed,
    )
    return response


# ---------------------------------------------------------------------------
# Dashboard view
# ---------------------------------------------------------------------------

@bp.route("/", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ORION"}), 200


# ---------------------------------------------------------------------------
# Upload data
# ---------------------------------------------------------------------------

@bp.route("/upload-data", methods=["POST"])
def upload_data():
    """
    Accept a CSV file and a table_type parameter (kpi | segment | event).
    Validates schema and stores the DataFrame in the in-memory cache.
    """
    if "file" not in request.files:
        return jsonify(*build_error_response("No file part in request."))

    file = request.files["file"]
    if file.filename == "":
        return jsonify(*build_error_response("No file selected."))

    table_type = sanitize_string(request.form.get("table_type", ""))
    if table_type not in ("kpi", "segment", "event"):
        return jsonify(*build_error_response("table_type must be one of: kpi, segment, event."))

    try:
        df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
    except Exception as exc:
        logger.exception("CSV parsing error.")
        return jsonify(*build_error_response(f"Failed to parse CSV: {exc}"))

    valid, error_msg = validate_dataframe(df, table_type)
    if not valid:
        return jsonify(*build_error_response(error_msg))

    try:
        df = coerce_dataframe(df, table_type)
    except Exception as exc:
        logger.exception("Type coercion error.")
        return jsonify(*build_error_response(f"Data type error: {exc}"))

    _cache[table_type] = df
    logger.info("Loaded %s table: %d rows.", table_type, len(df))

    return jsonify(build_success_response({
        "table_type": table_type,
        "rows": len(df),
        "columns": list(df.columns),
    }))


# ---------------------------------------------------------------------------
# Detect anomaly
# ---------------------------------------------------------------------------

@bp.route("/detect-anomaly", methods=["POST"])
def detect_anomaly():
    """
    Parameters (JSON body)
    ----------------------
    kpi_name : str   KPI to analyse.
    """
    body = request.get_json(silent=True) or {}
    kpi_name = sanitize_string(body.get("kpi_name", ""))

    if not kpi_name:
        return jsonify(*build_error_response("kpi_name is required."))

    if "kpi" not in _cache:
        return jsonify(*build_error_response("No KPI data loaded. Upload a KPI CSV first."))

    kpi_df = _cache["kpi"]
    filtered = kpi_df[kpi_df["kpi_name"] == kpi_name].sort_values("date")

    if filtered.empty:
        return jsonify(*build_error_response(f"KPI '{kpi_name}' not found in uploaded data."))

    series = filtered.set_index("date")["kpi_value"]

    try:
        detector = AnomalyDetector(model_store=_get_model_store())
        result_df = detector.detect(series)
    except Exception as exc:
        logger.exception("Anomaly detection failed.")
        return jsonify(*build_error_response(f"Anomaly detection error: {exc}", 500))

    anomalies = result_df[result_df["is_anomaly"]].reset_index()
    anomalies["date"] = anomalies["date"].dt.isoformat()

    return jsonify(build_success_response({
        "kpi_name": kpi_name,
        "total_points": len(result_df),
        "anomaly_count": int(result_df["is_anomaly"].sum()),
        "anomalies": anomalies[["date", "value", "anomaly_score"]].to_dict(orient="records"),
        "full_series": result_df.reset_index().assign(date=lambda d: d["date"].dt.isoformat()).to_dict(orient="records"),
    }))


# ---------------------------------------------------------------------------
# Root cause
# ---------------------------------------------------------------------------

@bp.route("/root-cause", methods=["POST"])
def root_cause():
    """
    Parameters (JSON body)
    ----------------------
    kpi_name      : str          KPI to explain.
    anomaly_date  : str (opt)    ISO date of the anomaly to explain.
    """
    body = request.get_json(silent=True) or {}
    kpi_name = sanitize_string(body.get("kpi_name", ""))
    anomaly_date = sanitize_string(body.get("anomaly_date", "")) or None

    if not kpi_name:
        return jsonify(*build_error_response("kpi_name is required."))

    for required_table in ("kpi", "segment"):
        if required_table not in _cache:
            return jsonify(*build_error_response(f"No {required_table} data loaded."))

    kpi_df = _cache["kpi"][_cache["kpi"]["kpi_name"] == kpi_name]
    segment_df = _cache["segment"]

    try:
        analyser = RootCauseAnalyser(model_store=_get_model_store())
        result = analyser.analyse(kpi_df, segment_df, anomaly_date=anomaly_date)
    except Exception as exc:
        logger.exception("Root cause analysis failed.")
        return jsonify(*build_error_response(f"Root cause analysis error: {exc}", 500))

    return jsonify(build_success_response(result))


# ---------------------------------------------------------------------------
# Simulate impact
# ---------------------------------------------------------------------------

@bp.route("/simulate-impact", methods=["POST"])
def simulate_impact():
    """
    Parameters (JSON body)
    ----------------------
    kpi_name      : str    KPI to project.
    segment_name  : str    Segment to perturb.
    change_pct    : float  Percentage change to apply to the segment.
    """
    body = request.get_json(silent=True) or {}
    kpi_name = sanitize_string(body.get("kpi_name", ""))
    segment_name = sanitize_string(body.get("segment_name", ""))

    try:
        change_pct = float(body.get("change_pct", 0))
    except (TypeError, ValueError):
        return jsonify(*build_error_response("change_pct must be a numeric value."))

    if not kpi_name or not segment_name:
        return jsonify(*build_error_response("kpi_name and segment_name are required."))

    for required_table in ("kpi", "segment"):
        if required_table not in _cache:
            return jsonify(*build_error_response(f"No {required_table} data loaded."))

    kpi_df = _cache["kpi"][_cache["kpi"]["kpi_name"] == kpi_name]
    segment_df = _cache["segment"]

    try:
        simulator = ImpactSimulator()
        result = simulator.simulate(kpi_df, segment_df, segment_name, change_pct)
    except ValueError as exc:
        return jsonify(*build_error_response(str(exc)))
    except Exception as exc:
        logger.exception("Impact simulation failed.")
        return jsonify(*build_error_response(f"Simulation error: {exc}", 500))

    return jsonify(build_success_response(result))


# ---------------------------------------------------------------------------
# Recommendation via Gemini API
# ---------------------------------------------------------------------------

@bp.route("/recommendation", methods=["POST"])
def recommendation():
    """
    Parameters (JSON body)
    ----------------------
    context : dict   Structured context from previous pipeline steps.
    """
    body = request.get_json(silent=True) or {}
    context = body.get("context", {})

    if not context:
        return jsonify(*build_error_response("context payload is required."))

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify(*build_error_response(
            "Gemini API key not configured. Set GEMINI_API_KEY in your environment.", 503
        ))

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = _build_recommendation_prompt(context)
        response = model.generate_content(prompt)
        recommendation_text = response.text

        logger.info("Gemini recommendation generated successfully.")
    except Exception as exc:
        logger.exception("Gemini API call failed.")
        return jsonify(*build_error_response(f"Recommendation generation failed: {exc}", 502))

    return jsonify(build_success_response({
        "recommendation": recommendation_text,
        "context_summary": context,
    }))


def _build_recommendation_prompt(context: dict) -> str:
    context_str = json.dumps(context, indent=2)
    return (
        "You are a senior business intelligence analyst. "
        "Based on the following KPI anomaly analysis context, provide specific, "
        "actionable operational recommendations. Structure your response with: "
        "(1) a brief situation summary, (2) identified root causes, "
        "(3) prioritised recommended actions with expected outcomes, "
        "and (4) monitoring guidance.\n\n"
        f"Context:\n{context_str}"
    )
