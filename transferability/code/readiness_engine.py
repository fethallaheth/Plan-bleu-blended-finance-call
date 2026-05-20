from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from .config import ALLOWED_COUNTRIES, COUNTRY_READINESS_INPUTS, READINESS_PILLARS, READINESS_WEIGHTS
except ImportError:
    from config import ALLOWED_COUNTRIES, COUNTRY_READINESS_INPUTS, READINESS_PILLARS, READINESS_WEIGHTS


def validate_readiness_inputs() -> None:
    configured_countries = set(COUNTRY_READINESS_INPUTS.keys())
    allowed = set(ALLOWED_COUNTRIES)
    if configured_countries != allowed:
        extra = configured_countries - allowed
        missing = allowed - configured_countries
        raise ValueError(f"Country scope mismatch. extra={sorted(extra)} missing={sorted(missing)}")

    for country, payload in COUNTRY_READINESS_INPUTS.items():
        if payload.get("source_type") != "DOCUMENTED_AUTHOR_SCORE":
            raise ValueError(f"{country} source_type must be DOCUMENTED_AUTHOR_SCORE in this version.")
        note = payload.get("main_note", "")
        if not note:
            raise ValueError(f"{country} must include main_note.")
        for pillar in READINESS_PILLARS:
            if pillar not in payload:
                raise ValueError(f"{country} missing pillar: {pillar}")

    weight_total = sum(READINESS_WEIGHTS.values())
    if not np.isclose(weight_total, 1.0):
        raise ValueError(f"Readiness weights must sum to 1.0, found {weight_total}.")


def classify_readiness(score: float) -> str:
    if score >= 70:
        return "High / near-term pilot"
    if score >= 55:
        return "Medium-high / adaptation needed"
    if score >= 40:
        return "Medium / capacity-building required"
    if score >= 25:
        return "Low-medium / foundation building first"
    return "Low / indirect or long-term only"


def build_country_readiness_scores() -> pd.DataFrame:
    validate_readiness_inputs()
    rows: list[dict[str, Any]] = []
    for country in ALLOWED_COUNTRIES:
        payload = COUNTRY_READINESS_INPUTS[country]
        score = float(sum(payload[pillar] * READINESS_WEIGHTS[pillar] for pillar in READINESS_PILLARS))
        weakest_pillar = min(READINESS_PILLARS, key=lambda p: payload[p])
        rows.append(
            {
                "country": country,
                **{pillar: float(payload[pillar]) for pillar in READINESS_PILLARS},
                "readiness_score": score,
                "readiness_classification": classify_readiness(score),
                "main_binding_constraint": weakest_pillar,
                "source_type": payload["source_type"],
                "main_note": payload["main_note"],
            }
        )
    return pd.DataFrame(rows)


def source_coverage_table(readiness_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in readiness_df.iterrows():
        rows.append(
            {
                "country": row["country"],
                "OBSERVED_DATA": 0.0,
                "DERIVED_DATA": 0.0,
                "DOCUMENTED_AUTHOR_SCORE": 100.0,
                "MODEL_ASSUMPTION": 0.0,
            }
        )
    return pd.DataFrame(rows)


def optional_dataset_validation_placeholder(external_indicator_map: dict[str, dict[str, float]] | None = None) -> str:
    if external_indicator_map is None:
        return "No external datasets supplied; structured readiness scores retained."
    return "External indicators supplied. Use a mapping pipeline to validate or replace structured scores."
