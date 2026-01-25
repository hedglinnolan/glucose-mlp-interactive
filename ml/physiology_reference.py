"""
Physiologic plausibility reference framework.

Empirical plausibility intervals are derived from NHANES-like population distributions
and are distinct from clinical guideline thresholds, which are informational only.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple, Any
from functools import lru_cache
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

from ml.clinical_units import CLINICAL_VARIABLES


DEFAULT_NHANES_REFERENCE: Dict[str, Any] = {
    "version": "nhanes_reference_demo_v1",
    "source": "NHANES (reference population, demo defaults)",
    "variables": {
        "glucose": {"unit": "mg/dL", "p01": 70, "p99": 200},
        "bmi": {"unit": "kg/m²", "p01": 15, "p99": 50},
        "hba1c": {"unit": "%", "p01": 4.0, "p99": 15.0},
        "cholesterol": {"unit": "mmol/L", "p01": 2.0, "p99": 10.0},
        "triglyceride": {"unit": "mg/dL", "p01": 50, "p99": 500},
        "weight": {"unit": "kg", "p01": 35, "p99": 200},
        "height": {"unit": "cm", "p01": 140, "p99": 210},
        "waist": {"unit": "cm", "p01": 55, "p99": 150},
        "bp_sys": {"unit": "mmHg", "p01": 90, "p99": 200},
        "bp_di": {"unit": "mmHg", "p01": 50, "p99": 120},
        "kcal": {"unit": "kcal", "p01": 800, "p99": 4500},
    },
}


def _build_clinical_guidelines() -> Dict[str, Any]:
    guidelines: Dict[str, Any] = {}
    for var_name, var_config in CLINICAL_VARIABLES.items():
        if "thresholds" in var_config:
            guidelines[var_name] = {
                "canonical_unit": var_config.get("canonical_unit"),
                "thresholds_by_unit": var_config.get("thresholds", {}),
                "fasting_note": var_config.get("fasting_note", False),
                "source": "Clinical guidelines (informational only)"
            }
    return guidelines


@lru_cache(maxsize=1)
def load_nhanes_reference(reference_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Load NHANES-based empirical plausibility reference.
    Attempts a dynamic pull when NHANES_REFERENCE_URL is set.
    Falls back to bundled defaults if unavailable.
    """
    url = reference_url or os.getenv("NHANES_REFERENCE_URL")
    if url:
        try:
            with urlopen(url, timeout=5) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)
                if _validate_reference(data):
                    return data
        except (URLError, HTTPError, ValueError):
            pass
    return DEFAULT_NHANES_REFERENCE


def load_reference_bundle(reference_url: Optional[str] = None) -> Dict[str, Any]:
    """Return both empirical NHANES reference and clinical guideline overlays."""
    return {
        "nhanes": load_nhanes_reference(reference_url),
        "clinical": _build_clinical_guidelines()
    }


def _validate_reference(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    if "variables" not in data:
        return False
    if not isinstance(data["variables"], dict) or len(data["variables"]) == 0:
        return False
    for _, payload in data["variables"].items():
        if not isinstance(payload, dict):
            return False
        if "unit" not in payload or "p01" not in payload or "p99" not in payload:
            return False
    return True


def match_variable_key(col_name: str, reference: Dict[str, Any]) -> Optional[str]:
    col_lower = col_name.lower()
    for key in reference.get("variables", {}).keys():
        if key in col_lower:
            return key
    return None


def get_reference_interval(reference: Dict[str, Any], var_key: str) -> Optional[Tuple[float, float, str]]:
    var_data = reference.get("variables", {}).get(var_key)
    if not var_data:
        return None
    return float(var_data["p01"]), float(var_data["p99"]), var_data["unit"]

