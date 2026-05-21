from __future__ import annotations

from pathlib import Path


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

READINESS_PILLARS = (
    "regulatory_clarity",
    "supervisory_capacity",
    "public_finance_compatibility",
    "legal_enforceability",
    "digital_infrastructure",
    "environmental_monitoring_capacity",
    "sustainable_finance_market_maturity",
)

READINESS_WEIGHTS = {
    "regulatory_clarity": 0.20,
    "supervisory_capacity": 0.15,
    "public_finance_compatibility": 0.15,
    "legal_enforceability": 0.15,
    "digital_infrastructure": 0.15,
    "environmental_monitoring_capacity": 0.10,
    "sustainable_finance_market_maturity": 0.10,
}

COUNTRY_READINESS_INPUTS = {
    "France": {
        "regulatory_clarity": 75,
        "supervisory_capacity": 80,
        "public_finance_compatibility": 80,
        "legal_enforceability": 65,
        "digital_infrastructure": 85,
        "environmental_monitoring_capacity": 85,
        "sustainable_finance_market_maturity": 80,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "main_note": "Benchmark pilot jurisdiction with strong EU/French institutional capacity but data and smart-contract enforceability constraints.",
    },
    "Italy": {
        "regulatory_clarity": 68,
        "supervisory_capacity": 70,
        "public_finance_compatibility": 65,
        "legal_enforceability": 62,
        "digital_infrastructure": 75,
        "environmental_monitoring_capacity": 78,
        "sustainable_finance_market_maturity": 68,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "main_note": "Near-term adaptation candidate with strong EU alignment and relevant Mediterranean coastal-monitoring context.",
    },
    "Spain": {
        "regulatory_clarity": 70,
        "supervisory_capacity": 72,
        "public_finance_compatibility": 68,
        "legal_enforceability": 65,
        "digital_infrastructure": 78,
        "environmental_monitoring_capacity": 80,
        "sustainable_finance_market_maturity": 70,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "main_note": "Near-term adaptation candidate with strong EU alignment and relatively strong digital/institutional capacity.",
    },
    "Greece": {
        "regulatory_clarity": 58,
        "supervisory_capacity": 60,
        "public_finance_compatibility": 55,
        "legal_enforceability": 55,
        "digital_infrastructure": 65,
        "environmental_monitoring_capacity": 70,
        "sustainable_finance_market_maturity": 55,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "main_note": "Medium adaptation candidate; coastal relevance is strong but public-finance and institutional capacity require support.",
    },
    "Tunisia": {
        "regulatory_clarity": 38,
        "supervisory_capacity": 42,
        "public_finance_compatibility": 35,
        "legal_enforceability": 40,
        "digital_infrastructure": 50,
        "environmental_monitoring_capacity": 45,
        "sustainable_finance_market_maturity": 35,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "main_note": "Longer-term transfer path requiring regulatory and sustainable-finance capacity building.",
    },
    "Egypt": {
        "regulatory_clarity": 45,
        "supervisory_capacity": 48,
        "public_finance_compatibility": 45,
        "legal_enforceability": 42,
        "digital_infrastructure": 55,
        "environmental_monitoring_capacity": 50,
        "sustainable_finance_market_maturity": 48,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "main_note": "Emerging adaptation case with sustainable-finance activity but institutional and legal enforceability constraints.",
    },
    "Turkey": {
        "regulatory_clarity": 52,
        "supervisory_capacity": 55,
        "public_finance_compatibility": 50,
        "legal_enforceability": 48,
        "digital_infrastructure": 68,
        "environmental_monitoring_capacity": 60,
        "sustainable_finance_market_maturity": 55,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "main_note": "Emerging market-infrastructure adaptation path with stronger digital base but regulatory and legal uncertainty.",
    },
    "Algeria": {
        "regulatory_clarity": 30,
        "supervisory_capacity": 35,
        "public_finance_compatibility": 32,
        "legal_enforceability": 35,
        "digital_infrastructure": 42,
        "environmental_monitoring_capacity": 40,
        "sustainable_finance_market_maturity": 25,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "main_note": "Long-term foundation-building transfer path; requires sustainable-finance, digital governance, and regulatory capacity development.",
    },
    "Lebanon": {
        "regulatory_clarity": 25,
        "supervisory_capacity": 25,
        "public_finance_compatibility": 20,
        "legal_enforceability": 25,
        "digital_infrastructure": 45,
        "environmental_monitoring_capacity": 40,
        "sustainable_finance_market_maturity": 20,
        "source_type": "DOCUMENTED_AUTHOR_SCORE",
        "main_note": "Long-term/indirect transfer case due to public-finance and institutional constraints.",
    },
}

SIMULATION = {
    "draws_per_country": 5000,
    "random_seed": 42,
    "source_type": "MODEL_ASSUMPTION",
    "basis": "Monte Carlo stabilization for country transferability estimates.",
}

TRANSFER_PARAMETERS = {
    "expected_mobilization_base": {
        "value": 4.30,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Baseline expected mobilization near benchmark readiness.",
    },
    "mobilization_gap_slope": {
        "value": 0.012,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Mobilization penalty for readiness gap versus France.",
    },
    "risk_premium_base_bps": {
        "value": 300.0,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Baseline risk premium benchmark in basis points.",
    },
    "risk_gap_slope_bps": {
        "value": 3.5,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Risk premium increase for readiness gap versus France.",
    },
    "constraint_risk_slope_bps": {
        "value": 1.5,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Incremental risk premium per weakest-pillar point below the 60-point threshold.",
    },
    "issuance_sigmoid_slope": {
        "value": 0.12,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Sigmoid slope for converting readiness scores into issuance feasibility.",
    },
    "adaptation_months_base": {
        "value": 9.0,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Base preparation time before adaptation penalties.",
    },
    "adaptation_gap_month_slope": {
        "value": 0.8,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Additional adaptation months per readiness gap point.",
    },
    "mobilization_noise_sd": {
        "value": 0.05,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Country-level uncertainty for expected mobilization.",
    },
    "constraint_mobilization_slope": {
        "value": 0.008,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Mobilization penalty per weakest-pillar point below the 60-point threshold.",
    },
    "risk_noise_sd": {
        "value": 8.0,
        "source_type": "MODEL_ASSUMPTION",
        "basis": "Country-level uncertainty for expected risk premium.",
    },
}

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
