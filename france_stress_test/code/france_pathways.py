from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from .config import FRANCE_PATHWAY_SCORES, FRANCE_STRESS_SCENARIOS, PATHWAY_STRESS_SENSITIVITY
except ImportError:
    from config import FRANCE_PATHWAY_SCORES, FRANCE_STRESS_SCENARIOS, PATHWAY_STRESS_SENSITIVITY


@dataclass(frozen=True)
class ReadinessResult:
    score: float
    classification: str


def build_pathway_dataframe() -> pd.DataFrame:
    records = []
    for pathway, payload in FRANCE_PATHWAY_SCORES.items():
        records.append(
            {
                "pathway": pathway,
                "score": float(payload["score"]),
                "weight": float(payload["weight"]),
                "weighted_score": float(payload["score"]) * float(payload["weight"]),
                "source_type": payload["source_type"],
                "basis": payload["basis"],
            }
        )
    frame = pd.DataFrame(records)
    frame["score_gap_to_threshold_70"] = frame["score"] - 70.0
    return frame


def classify_readiness(score: float) -> str:
    if score >= 70.0:
        return "Ready for pilot"
    if score >= 55.0:
        return "Conditional"
    return "Not ready"


def compute_france_readiness(pathway_df: pd.DataFrame) -> ReadinessResult:
    score = float(pathway_df["weighted_score"].sum())
    return ReadinessResult(score=score, classification=classify_readiness(score))


def compute_scenario_pathway_integrities(scenario_key: str) -> dict[str, float]:
    scenario = FRANCE_STRESS_SCENARIOS[scenario_key]
    base_scores = {name: float(payload["score"]) / 100.0 for name, payload in FRANCE_PATHWAY_SCORES.items()}
    integrities: dict[str, float] = {}
    delay_total = (
        float(scenario["regulatory_delay_months"])
        + float(scenario["legal_delay_months"])
        + float(scenario["validation_delay_months"])
        + float(scenario["settlement_delay_months"])
    )
    delay_scaled = min(delay_total / 60.0, 1.0)

    for pathway, sensitivities in PATHWAY_STRESS_SENSITIVITY.items():
        stress_penalty = (
            float(sensitivities["regulatory"]) * float(scenario["regulatory_delay_months"]) / 24.0
            + float(sensitivities["legal"]) * float(scenario["legal_delay_months"]) / 24.0
            + float(sensitivities["validation"]) * float(scenario["validation_delay_months"]) / 24.0
            + float(sensitivities["settlement"]) * float(scenario["settlement_delay_months"]) / 36.0
            + float(sensitivities["oracle"]) * float(scenario["oracle_challenge_severity"])
            + 0.15 * delay_scaled
        )
        integrities[pathway] = float(np.clip(base_scores[pathway] * (1.0 - stress_penalty), 0.0, 1.0))
    return integrities


def build_pathway_integrity_heatmap() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    scenario_integrities = {
        scenario_key: compute_scenario_pathway_integrities(scenario_key) for scenario_key in FRANCE_STRESS_SCENARIOS
    }
    for pathway in PATHWAY_STRESS_SENSITIVITY:
        row = {"pathway": pathway}
        for scenario_key, integrities in scenario_integrities.items():
            row[scenario_key] = integrities[pathway] * 100.0
        rows.append(row)
    return pd.DataFrame(rows)
