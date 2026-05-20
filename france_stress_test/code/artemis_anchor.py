from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from .config import ARTEMIS_ANCHOR
except ImportError:
    from config import ARTEMIS_ANCHOR


def _ensure_base_module_path() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.append(str(root))


@dataclass(frozen=True)
class ArtemisBaseline:
    mobilization_ratio_median: float
    risk_premium_bps_median: float
    governance_cost_pct_median: float
    disbursement_lag_months_median: float
    source_type: str
    basis: str


def load_blockchain_baseline() -> ArtemisBaseline:
    _ensure_base_module_path()
    from blue_bond_simulation.config import DEFAULT_CONFIG  # pylint: disable=import-outside-toplevel
    from blue_bond_simulation.engine import BlueBondEngine  # pylint: disable=import-outside-toplevel

    engine = BlueBondEngine(DEFAULT_CONFIG)
    results = engine.run()
    scenario_name = DEFAULT_CONFIG["scenario_b"]["name"]
    scenario_df = results[results["scenario"] == scenario_name]
    return ArtemisBaseline(
        mobilization_ratio_median=float(np.median(scenario_df["mobilization_ratio"])),
        risk_premium_bps_median=float(np.median(scenario_df["risk_premium_bps"])),
        governance_cost_pct_median=float(np.median(scenario_df["total_governance_cost_pct"])),
        disbursement_lag_months_median=float(np.median(scenario_df["disbursement_lag_months"])),
        source_type="DERIVED_DATA",
        basis="Derived from the existing blockchain-enabled scenario in the base comparative Monte Carlo model.",
    )


def anchor_table() -> pd.DataFrame:
    baseline = load_blockchain_baseline()
    rows = []
    for metric, payload in ARTEMIS_ANCHOR.items():
        rows.append(
            {
                "parameter": metric,
                "value": payload["value"],
                "source_type": payload["source_type"],
                "basis": payload["basis"],
            }
        )
    rows.extend(
        [
            {
                "parameter": "base_model_blockchain_mobilization_ratio_median",
                "value": baseline.mobilization_ratio_median,
                "source_type": baseline.source_type,
                "basis": baseline.basis,
            },
            {
                "parameter": "base_model_blockchain_risk_premium_bps_median",
                "value": baseline.risk_premium_bps_median,
                "source_type": baseline.source_type,
                "basis": baseline.basis,
            },
            {
                "parameter": "base_model_blockchain_governance_cost_pct_median",
                "value": baseline.governance_cost_pct_median,
                "source_type": baseline.source_type,
                "basis": baseline.basis,
            },
            {
                "parameter": "base_model_blockchain_disbursement_lag_months_median",
                "value": baseline.disbursement_lag_months_median,
                "source_type": baseline.source_type,
                "basis": baseline.basis,
            },
        ]
    )
    return pd.DataFrame(rows)
