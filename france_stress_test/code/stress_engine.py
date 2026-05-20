from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from .config import ARTEMIS_ANCHOR, FRANCE_PATHWAY_SCORES, FRANCE_STRESS_SCENARIOS, MODEL_PARAMETERS, SIMULATION, sigmoid
except ImportError:
    from config import ARTEMIS_ANCHOR, FRANCE_PATHWAY_SCORES, FRANCE_STRESS_SCENARIOS, MODEL_PARAMETERS, SIMULATION, sigmoid


@dataclass(frozen=True)
class StressRunResult:
    iterations: pd.DataFrame
    summary: pd.DataFrame


def _require_source_fields(payload: dict, context: str) -> None:
    if "source_type" not in payload:
        raise ValueError(f"{context} must define source_type.")
    if payload["source_type"] not in {"OBSERVED_DATA", "DERIVED_DATA", "DOCUMENTED_AUTHOR_SCORE", "MODEL_ASSUMPTION"}:
        raise ValueError(f"{context} has invalid source_type: {payload['source_type']}")
    if context.startswith("scenario:"):
        if "narrative" not in payload:
            raise ValueError(f"{context} must define narrative.")
    elif "basis" not in payload:
        raise ValueError(f"{context} must define basis.")


def validate_stress_inputs() -> None:
    for pathway_name, payload in FRANCE_PATHWAY_SCORES.items():
        _require_source_fields(payload, f"pathway:{pathway_name}")
    for anchor_name, payload in ARTEMIS_ANCHOR.items():
        _require_source_fields(payload, f"anchor:{anchor_name}")
    for parameter_name, payload in MODEL_PARAMETERS.items():
        _require_source_fields(payload, f"model_parameter:{parameter_name}")
    for scenario_name, payload in FRANCE_STRESS_SCENARIOS.items():
        _require_source_fields(payload, f"scenario:{scenario_name}")


def _status_from_metrics(issuance_probability: np.ndarray, admissibility: np.ndarray, time_months: np.ndarray) -> np.ndarray:
    status = np.full(issuance_probability.shape[0], "FAIL", dtype=object)
    pass_mask = (issuance_probability >= 0.70) & (admissibility >= 0.60) & (time_months <= 30.0)
    conditional_mask = (~pass_mask) & (issuance_probability >= 0.40) & (admissibility >= 0.30)
    status[conditional_mask] = "CONDITIONAL"
    status[pass_mask] = "PASS"
    return status


def run_france_stress_test() -> StressRunResult:
    validate_stress_inputs()
    iterations = int(SIMULATION["iterations"])
    random_seed = int(SIMULATION["random_seed"])
    cnil_score = float(FRANCE_PATHWAY_SCORES["CNIL / GDPR"]["score"]) / 100.0
    params = {name: float(payload["value"]) for name, payload in MODEL_PARAMETERS.items()}
    anchor = {name: float(payload["value"]) for name, payload in ARTEMIS_ANCHOR.items()}

    frames: list[pd.DataFrame] = []
    for index, (scenario_key, scenario) in enumerate(FRANCE_STRESS_SCENARIOS.items()):
        rng = np.random.default_rng(random_seed + index)
        france_integrity = float(scenario["france_integrity"])
        artemis_integrity = float(scenario["artemis_integrity"])
        institutional_quality = min(france_integrity, artemis_integrity)
        average_quality = 0.5 * (france_integrity + artemis_integrity)

        risk_premium = (
            anchor["base_risk_premium_bps"]
            + float(scenario["risk_shock_bps"])
            + float(scenario["oracle_challenge_severity"]) * params["oracle_risk_premium_multiplier"]
            + rng.normal(0.0, params["risk_noise_sd"], iterations)
        )
        risk_premium = np.clip(risk_premium, 0.0, None)

        governance_cost = (
            anchor["base_governance_cost_pct"]
            + float(scenario["governance_cost_shock"])
            + float(scenario["oracle_challenge_severity"]) * params["oracle_cost_multiplier"]
            + rng.normal(0.0, params["governance_noise_sd"], iterations)
        )
        governance_cost = np.clip(governance_cost, 0.0, None)

        mobilization_ratio = (
            anchor["base_mobilization_ratio"]
            + anchor["blockchain_mobilization_premium"] * institutional_quality
            - (1.0 - institutional_quality) * params["mobilization_institutional_penalty_weight"]
            - (risk_premium - anchor["base_risk_premium_bps"]) / params["risk_premium_bps_scale"]
            + rng.normal(0.0, params["mobilization_noise_sd"], iterations)
        )
        mobilization_ratio = np.clip(mobilization_ratio, 0.0, None)

        total_delay = (
            float(scenario["regulatory_delay_months"])
            + float(scenario["legal_delay_months"])
            + float(scenario["validation_delay_months"])
            + float(scenario["settlement_delay_months"])
        )
        legal_validation_factor = np.clip(
            1.0 - total_delay / params["legal_validation_decay_months"],
            params["legal_validation_floor"],
            1.0,
        )
        admissibility = (
            artemis_integrity
            * cnil_score
            * (1.0 - float(scenario["oracle_challenge_severity"]))
            * legal_validation_factor
        )
        admissibility = np.clip(admissibility, 0.0, 1.0)
        admissibility = np.full(iterations, admissibility)

        issuance_probability = sigmoid((institutional_quality - 0.50) * params["issuance_sigmoid_scale"])
        issuance_probability = np.clip(issuance_probability, 0.0, 1.0)
        issuance_probability = np.full(iterations, issuance_probability)

        pes_random_factor = np.clip(rng.normal(1.0, params["pes_random_sd"], iterations), 0.0, None)
        pes_achievement = (
            anchor["pes_target_eur"] * artemis_integrity * issuance_probability * pes_random_factor
        )
        pes_achievement = np.clip(pes_achievement, 0.0, None)

        time_to_issuance = (
            anchor["base_time_to_issuance_months"]
            + float(scenario["regulatory_delay_months"])
            + float(scenario["legal_delay_months"])
            + float(scenario["validation_delay_months"])
            + float(scenario["settlement_delay_months"])
            + rng.normal(0.0, params["time_noise_sd"], iterations)
        )
        time_to_issuance = np.clip(time_to_issuance, 0.0, None)

        status = _status_from_metrics(issuance_probability, admissibility, time_to_issuance)
        frames.append(
            pd.DataFrame(
                {
                    "scenario_key": scenario_key,
                    "scenario_label": scenario["label"],
                    "iteration": np.arange(1, iterations + 1),
                    "france_integrity": france_integrity,
                    "artemis_integrity": artemis_integrity,
                    "institutional_quality": institutional_quality,
                    "average_quality": average_quality,
                    "mobilization_ratio": mobilization_ratio,
                    "risk_premium_bps": risk_premium,
                    "governance_cost_pct": governance_cost,
                    "pes_achievement_eur": pes_achievement,
                    "data_admissibility_probability": admissibility,
                    "issuance_probability": issuance_probability,
                    "time_to_issuance_months": time_to_issuance,
                    "status": status,
                    "target_status": scenario["target_status"],
                    "source_type": "DERIVED_DATA",
                }
            )
        )

    iterations_df = pd.concat(frames, ignore_index=True)
    summary_df = summarize_iterations(iterations_df)
    return StressRunResult(iterations=iterations_df, summary=summary_df)


def summarize_iterations(iterations_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario_label, scenario_df in iterations_df.groupby("scenario_label", sort=False):
        status_mode = scenario_df["status"].value_counts().idxmax()
        scenario_key = scenario_df["scenario_key"].iloc[0]
        rows.append(
            {
                "scenario_key": scenario_key,
                "scenario_label": scenario_label,
                "france_integrity": float(scenario_df["france_integrity"].median()),
                "artemis_integrity": float(scenario_df["artemis_integrity"].median()),
                "institutional_quality": float(scenario_df["institutional_quality"].median()),
                "average_quality": float(scenario_df["average_quality"].median()),
                "median_mobilization_ratio": float(scenario_df["mobilization_ratio"].median()),
                "median_risk_premium_bps": float(scenario_df["risk_premium_bps"].median()),
                "risk_premium_p10": float(np.percentile(scenario_df["risk_premium_bps"], 10)),
                "risk_premium_p90": float(np.percentile(scenario_df["risk_premium_bps"], 90)),
                "median_governance_cost_pct": float(scenario_df["governance_cost_pct"].median()),
                "governance_cost_p10": float(np.percentile(scenario_df["governance_cost_pct"], 10)),
                "governance_cost_p90": float(np.percentile(scenario_df["governance_cost_pct"], 90)),
                "median_pes_achievement_eur": float(scenario_df["pes_achievement_eur"].median()),
                "median_data_admissibility_probability": float(
                    scenario_df["data_admissibility_probability"].median()
                ),
                "median_issuance_probability": float(scenario_df["issuance_probability"].median()),
                "median_time_to_issuance_months": float(scenario_df["time_to_issuance_months"].median()),
                "status": status_mode,
                "target_status": scenario_df["target_status"].iloc[0],
                "source_type": "DERIVED_DATA",
            }
        )
    return pd.DataFrame(rows)
