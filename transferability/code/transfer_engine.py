from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from .config import READINESS_PILLARS, SIMULATION, TRANSFER_PARAMETERS
except ImportError:
    from config import READINESS_PILLARS, SIMULATION, TRANSFER_PARAMETERS


@dataclass(frozen=True)
class TransferResult:
    readiness_scores: pd.DataFrame
    transferability: pd.DataFrame


def _constraint_penalty(lowest_pillar_value: float) -> float:
    if lowest_pillar_value < 25:
        return 24.0
    if lowest_pillar_value < 40:
        return 12.0
    if lowest_pillar_value < 55:
        return 6.0
    return 0.0


def _transferability_grade(score: float) -> str:
    if score >= 70:
        return "Near-term transferable"
    if score >= 55:
        return "Transferable with adaptation"
    if score >= 40:
        return "Capacity-building before transfer"
    if score >= 25:
        return "Foundation-building first"
    return "Indirect / very long-term transfer"


def _adaptation_track(months: float) -> str:
    if months <= 24:
        return "near-term"
    if months <= 48:
        return "medium-term"
    if months <= 72:
        return "long-term"
    return "indirect/very long-term"


def run_transferability_model(readiness_df: pd.DataFrame) -> TransferResult:
    params = {name: float(payload["value"]) for name, payload in TRANSFER_PARAMETERS.items()}
    draws = int(SIMULATION["draws_per_country"])
    rng = np.random.default_rng(int(SIMULATION["random_seed"]))

    france_score = float(readiness_df.loc[readiness_df["country"] == "France", "readiness_score"].iloc[0])
    rows: list[dict[str, float | str]] = []
    for _, country_row in readiness_df.iterrows():
        country = str(country_row["country"])
        country_score = float(country_row["readiness_score"])
        readiness_gap = france_score - country_score

        mobilization_draws = (
            params["expected_mobilization_base"]
            - readiness_gap * params["mobilization_gap_slope"]
            + rng.normal(0.0, params["mobilization_noise_sd"], draws)
        )
        mobilization_draws = np.clip(mobilization_draws, 0.0, None)
        expected_mobilization = float(np.mean(mobilization_draws))

        risk_draws = (
            params["risk_premium_base_bps"]
            + readiness_gap * params["risk_gap_slope_bps"]
            + rng.normal(0.0, params["risk_noise_sd"], draws)
        )
        risk_draws = np.clip(risk_draws, 0.0, None)
        expected_risk_premium = float(np.mean(risk_draws))

        issuance_probability = float(1.0 / (1.0 + np.exp(-((country_score - 50.0) * params["issuance_sigmoid_slope"]))))
        lowest_pillar = str(country_row["main_binding_constraint"])
        lowest_pillar_value = float(country_row[lowest_pillar])
        penalty = _constraint_penalty(lowest_pillar_value)
        adaptation_time = (
            params["adaptation_months_base"]
            + max(readiness_gap, 0.0) * params["adaptation_gap_month_slope"]
            + penalty
        )
        rows.append(
            {
                "country": country,
                "readiness_score": country_score,
                "readiness_gap_vs_france": readiness_gap,
                "expected_mobilization_ratio": expected_mobilization,
                "expected_risk_premium_bps": expected_risk_premium,
                "issuance_probability": issuance_probability,
                "adaptation_time_months": adaptation_time,
                "main_binding_constraint": lowest_pillar,
                "lowest_pillar_score": lowest_pillar_value,
                "transferability_grade": _transferability_grade(country_score),
                "adaptation_track": _adaptation_track(adaptation_time),
                "source_type": "DERIVED_DATA",
            }
        )

    result_df = pd.DataFrame(rows).sort_values("readiness_score", ascending=False).reset_index(drop=True)
    return TransferResult(readiness_scores=readiness_df, transferability=result_df)


def build_readiness_matrix(readiness_df: pd.DataFrame) -> pd.DataFrame:
    return readiness_df[["country", *READINESS_PILLARS, "readiness_score"]].copy()
