from __future__ import annotations

from pathlib import Path

import numpy as np


ALLOWED_COUNTRIES = (
    "France",
    "Italy",
    "Spain",
    "Greece",
    "Tunisia",
    "Egypt",
    "Turkey",
    "Algeria",
    "Lebanon",
)

SOURCE_TYPES = (
    "OBSERVED_DATA",
    "DERIVED_DATA",
    "DOCUMENTED_AUTHOR_SCORE",
    "MODEL_ASSUMPTION",
)

SIMULATION = {
    "iterations": 10_000,
    "random_seed": 42,
    "source_type": "MODEL_ASSUMPTION",
    "basis": "Simulation control settings for stress testing.",
}

FRANCE_PATHWAY_SCORES = {
    "AMF Bond Issuance": {
        "score": 75,
        "weight": 0.20,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "basis": "Structured institutional score used in the paper; reflects French capital-market supervision and uncertainty around blockchain governance-layer classification.",
    },
    "ACPR AML/KYC": {
        "score": 70,
        "weight": 0.18,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "basis": "Structured institutional score used in the paper; reflects AML/KYC supervisory capacity and uncertainty around ZKP compliance attestations.",
    },
    "Banque de France / ECB Settlement": {
        "score": 85,
        "weight": 0.15,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "basis": "Structured institutional score used in the paper; reflects strong euro settlement infrastructure but limited production use of DLT settlement for this structure.",
    },
    "CMF + Civil Code": {
        "score": 65,
        "weight": 0.17,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "basis": "Structured institutional score used in the paper; reflects uncertainty around smart-contract escrow enforceability and insolvency treatment.",
    },
    "AFT / LOLF Public Finance": {
        "score": 80,
        "weight": 0.15,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "basis": "Structured institutional score used in the paper; reflects strong public-finance capacity but possible friction with programmable disbursement.",
    },
    "CNIL / GDPR": {
        "score": 55,
        "weight": 0.15,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "basis": "Structured institutional score used in the paper; reflects GDPR, data immutability, and ZKP admissibility uncertainty.",
    },
}

ARTEMIS_ANCHOR = {
    "pes_target_eur": {
        "value": 2_000_000,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "basis": "PES mobilization target used in the paper as the post-ARTEMIS scaling benchmark.",
    },
    "restoration_scope_hectares": {
        "value": 50,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Scaled post-ARTEMIS restoration scope for simulation.",
    },
    "ecosystem_service_value_eur_per_ha_year": {
        "value": 86_000,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "basis": "Value used in the paper for ecosystem service valuation anchoring.",
    },
    "first_loss_tranche": {
        "value": 0.15,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Assumed first-loss/public guarantee share in the simulated blended Blue Bond.",
    },
    "senior_private_tranche": {
        "value": 0.85,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Assumed senior private tranche share.",
    },
    "base_time_to_issuance_months": {
        "value": 9,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Baseline issuance preparation period before stress delays.",
    },
    "base_mobilization_ratio": {
        "value": 4.0,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Baseline catalytic mobilization target.",
    },
    "blockchain_mobilization_premium": {
        "value": 0.30,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "basis": "Median uplift used in the paper from blockchain-enabled governance.",
    },
    "base_risk_premium_bps": {
        "value": 300,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Baseline senior-tranche risk premium assumption.",
    },
    "base_governance_cost_pct": {
        "value": 0.035,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Baseline governance cost as percentage of project value.",
    },
}

FRANCE_STRESS_SCENARIOS = {
    "baseline": {
        "label": "Baseline",
        "france_integrity": 1.00,
        "artemis_integrity": 1.00,
        "risk_shock_bps": 0,
        "governance_cost_shock": 0.000,
        "regulatory_delay_months": 1.5,
        "legal_delay_months": 0,
        "validation_delay_months": 0,
        "settlement_delay_months": 0,
        "oracle_challenge_severity": 0.00,
        "target_status": "PASS",
        "source_type": "MODEL_ASSUMPTION",
        "narrative": "Normal supervision; agency model works; ARTEMIS protocols operational.",
    },
    "regulatory_classification": {
        "label": "A: Regulatory Classification Challenge",
        "france_integrity": 0.83,
        "artemis_integrity": 0.80,
        "risk_shock_bps": 70,
        "governance_cost_shock": 0.006,
        "regulatory_delay_months": 14,
        "legal_delay_months": 2,
        "validation_delay_months": 0,
        "settlement_delay_months": 0,
        "oracle_challenge_severity": 0.10,
        "target_status": "CONDITIONAL",
        "source_type": "MODEL_ASSUMPTION",
        "narrative": "AMF review delay and classification uncertainty around the blockchain governance layer.",
    },
    "greenwashing_oracle_challenge": {
        "label": "B: Greenwashing / Oracle Data Challenge",
        "france_integrity": 0.73,
        "artemis_integrity": 0.45,
        "risk_shock_bps": 200,
        "governance_cost_shock": 0.018,
        "regulatory_delay_months": 18,
        "legal_delay_months": 24,
        "validation_delay_months": 24,
        "settlement_delay_months": 2,
        "oracle_challenge_severity": 0.75,
        "target_status": "FAIL",
        "source_type": "MODEL_ASSUMPTION",
        "narrative": "Greenwashing allegation, oracle data challenge, expert review, and uncertainty around ZKP/data admissibility.",
    },
    "public_finance_settlement_stress": {
        "label": "C: Public Finance / Settlement Stress",
        "france_integrity": 0.66,
        "artemis_integrity": 0.65,
        "risk_shock_bps": 125,
        "governance_cost_shock": 0.012,
        "regulatory_delay_months": 12,
        "legal_delay_months": 14,
        "validation_delay_months": 6,
        "settlement_delay_months": 36,
        "oracle_challenge_severity": 0.25,
        "target_status": "FAIL",
        "source_type": "MODEL_ASSUMPTION",
        "narrative": "Sovereign/public-finance pressure, agency issuance difficulty, and settlement infrastructure stress.",
    },
    "combined_systemic_stress": {
        "label": "D: Combined Systemic Stress",
        "france_integrity": 0.30,
        "artemis_integrity": 0.24,
        "risk_shock_bps": 275,
        "governance_cost_shock": 0.030,
        "regulatory_delay_months": 24,
        "legal_delay_months": 24,
        "validation_delay_months": 24,
        "settlement_delay_months": 16,
        "oracle_challenge_severity": 0.90,
        "target_status": "FAIL",
        "source_type": "MODEL_ASSUMPTION",
        "narrative": "Regulatory classification failure, ZKP rejection, guarantee failure, and operational disruption occur together.",
    },
}

MODEL_PARAMETERS = {
    "mobilization_institutional_penalty_weight": {
        "value": 0.35,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Penalty scaling on mobilization from institutional stress.",
    },
    "risk_premium_bps_scale": {
        "value": 1000,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Scale factor converting risk premium deltas into mobilization penalty.",
    },
    "oracle_risk_premium_multiplier": {
        "value": 80,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Additional risk premium from data/oracle challenge severity.",
    },
    "oracle_cost_multiplier": {
        "value": 0.006,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Additional governance cost from oracle challenge severity.",
    },
    "legal_validation_floor": {
        "value": 0.05,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Minimum legal validation factor under extreme delay.",
    },
    "legal_validation_decay_months": {
        "value": 48,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Validation quality decay denominator across total delay.",
    },
    "mobilization_noise_sd": {
        "value": 0.06,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Monte Carlo noise parameter for mobilization.",
    },
    "risk_noise_sd": {
        "value": 10.0,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Monte Carlo noise parameter for risk premium.",
    },
    "governance_noise_sd": {
        "value": 0.002,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Monte Carlo noise parameter for governance cost.",
    },
    "pes_random_sd": {
        "value": 0.08,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Monte Carlo multiplicative uncertainty for PES achievement.",
    },
    "time_noise_sd": {
        "value": 1.5,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Monte Carlo noise parameter for time to issuance in months.",
    },
    "admissibility_noise_sd": {
        "value": 0.035,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Monte Carlo uncertainty around legal/data admissibility after pathway stress is applied.",
    },
    "issuance_noise_sd": {
        "value": 0.35,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Monte Carlo uncertainty on the issuance-feasibility latent score.",
    },
    "admissibility_oracle_penalty_weight": {
        "value": 0.35,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Additional admissibility penalty applied to scenario oracle challenge severity.",
    },
    "issuance_sigmoid_scale": {
        "value": 10.0,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Slope of issuance feasibility sigmoid around 0.50 integrity.",
    },
}

PATHWAY_STRESS_SENSITIVITY = {
    "AMF Bond Issuance": {
        "regulatory": 0.45,
        "legal": 0.10,
        "validation": 0.10,
        "settlement": 0.05,
        "oracle": 0.30,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Regulatory pathway is most sensitive to review and classification delays.",
    },
    "ACPR AML/KYC": {
        "regulatory": 0.35,
        "legal": 0.15,
        "validation": 0.20,
        "settlement": 0.05,
        "oracle": 0.25,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "AML/KYC pathway is sensitive to both regulatory and validation pressure.",
    },
    "Banque de France / ECB Settlement": {
        "regulatory": 0.15,
        "legal": 0.10,
        "validation": 0.10,
        "settlement": 0.55,
        "oracle": 0.10,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Settlement pathway is primarily sensitive to settlement disruption.",
    },
    "CMF + Civil Code": {
        "regulatory": 0.20,
        "legal": 0.45,
        "validation": 0.20,
        "settlement": 0.05,
        "oracle": 0.10,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Legal pathway is highly sensitive to legal and validation burden.",
    },
    "AFT / LOLF Public Finance": {
        "regulatory": 0.25,
        "legal": 0.20,
        "validation": 0.10,
        "settlement": 0.35,
        "oracle": 0.10,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Public-finance pathway depends on settlement and legal integration.",
    },
    "CNIL / GDPR": {
        "regulatory": 0.15,
        "legal": 0.25,
        "validation": 0.25,
        "settlement": 0.05,
        "oracle": 0.30,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Data-governance pathway is highly sensitive to oracle disputes and legal validation.",
    },
}

CHART_STYLE = {
    "dpi": 300,
    "formats": ("png", "svg"),
    "palette": ["#1f3b73", "#2e6f95", "#6ca6c1", "#a2d2ff", "#4c956c"],
    "source_type": "MODEL_ASSUMPTION",
    "basis": "Presentation defaults for the stress-test chart pack.",
}

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
AUDIT_DIR = OUTPUT_DIR / "audit"
REPORT_DIR = OUTPUT_DIR / "report"


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-values))
