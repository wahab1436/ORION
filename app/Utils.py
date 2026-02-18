import logging
import time
import functools
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required column schemas for each table type
# ---------------------------------------------------------------------------

KPI_SCHEMA = {
    "date": "datetime64[ns]",
    "kpi_name": "object",
    "kpi_value": "float64",
    "target_value": "float64",
}

SEGMENT_SCHEMA = {
    "date": "datetime64[ns]",
    "segment_type": "object",
    "segment_name": "object",
    "segment_kpi_value": "float64",
    "segment_change": "float64",
    "weight": "float64",
}

EVENT_SCHEMA = {
    "date": "datetime64[ns]",
    "event_type": "object",
    "event_name": "object",
    "magnitude": "float64",
}

REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "kpi": list(KPI_SCHEMA.keys()),
    "segment": list(SEGMENT_SCHEMA.keys()),
    "event": list(EVENT_SCHEMA.keys()),
}


def validate_dataframe(df: pd.DataFrame, table_type: str) -> Tuple[bool, str]:
    """
    Validate that a DataFrame conforms to the expected schema.

    Returns (True, "") on success, (False, error_message) on failure.
    """
    required = REQUIRED_COLUMNS.get(table_type)
    if required is None:
        return False, f"Unknown table type: {table_type}"

    missing = [col for col in required if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"

    if df.empty:
        return False, "Uploaded file contains no data rows."

    return True, ""


def coerce_dataframe(df: pd.DataFrame, table_type: str) -> pd.DataFrame:
    """
    Attempt to coerce DataFrame columns to their expected types.
    Raises ValueError if coercion fails.
    """
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="raise")

    numeric_columns = {
        "kpi": ["kpi_value", "target_value"],
        "segment": ["segment_kpi_value", "segment_change", "weight"],
        "event": ["magnitude"],
    }

    for col in numeric_columns.get(table_type, []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def sanitize_string(value: Any, max_length: int = 256) -> str:
    """Strip and truncate a string value to prevent oversized inputs."""
    return str(value).strip()[:max_length]


def timing(func):
    """Decorator to log execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug("Function '%s' executed in %.4f seconds.", func.__name__, elapsed)
        return result
    return wrapper


def safe_json(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas types to JSON-serialisable Python types.
    """
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def build_error_response(message: str, status: int = 400) -> Tuple[Dict, int]:
    """Return a standardised JSON error envelope."""
    logger.warning("Error response (%d): %s", status, message)
    return {"success": False, "error": message}, status


def build_success_response(data: Any) -> Dict:
    """Return a standardised JSON success envelope."""
    return {"success": True, "data": safe_json(data)}
