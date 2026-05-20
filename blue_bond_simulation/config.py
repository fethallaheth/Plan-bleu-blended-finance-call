from __future__ import annotations

import copy
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "simulation": {
        "iterations": 10000,
        "random_seed": 42,
        "project_value_euro": 100000000,
    },
    "scenario_a": {
        "name": "Traditional Governance",
        "monitoring_cost_pct": {"min": 0.02, "mode": 0.035, "max": 0.05},
        "disbursement_lag_months": {"min": 3, "mode": 4.5, "max": 6},
        "mobilization_ratio": {"min": 3.2, "mode": 4.0, "max": 4.8},
        "risk_premium_bps": {"min": 250, "mode": 300, "max": 350},
    },
    "scenario_b": {
        "name": "Blockchain-Enabled Governance",
        "mrv_reduction": {"min": 0.10, "mode": 0.20, "max": 0.30},
        "lag_reduction": {"min": 0.20, "mode": 0.30, "max": 0.40},
        "risk_premium_reduction_bps": {"min": 10, "mode": 30, "max": 50},
        "bluewashing_risk_reduction": {"min": 0.15, "mode": 0.20, "max": 0.25},
        "implementation_cost_pct": {"min": 0.002, "mode": 0.005, "max": 0.010},
        "legal_uncertainty_penalty": {"min": 0.00, "mode": 0.03, "max": 0.08},
        "oracle_failure_risk": {"min": 0.00, "mode": 0.05, "max": 0.15},
        "adoption_friction": {"min": 0.00, "mode": 0.05, "max": 0.12},
    },
    "mobilization_model": {
        "literature_baseline": 4.0,
        "mrv_weight": 0.25,
        "lag_weight": 0.20,
        "premium_weight": 0.30,
        "bluewashing_weight": 0.25,
        "max_mobilization_ratio": 7.0,
    },
    "charting": {
        "output_dir": "outputs",
        "figure_dpi": 300,
        "figure_format": "png",
        "artemis_note": "ARTEMIS reference: EUR3M total budget, 80% publicly co-financed",
    },
}


TRIANGULAR_PATHS = (
    ("scenario_a", "monitoring_cost_pct"),
    ("scenario_a", "disbursement_lag_months"),
    ("scenario_a", "mobilization_ratio"),
    ("scenario_a", "risk_premium_bps"),
    ("scenario_b", "mrv_reduction"),
    ("scenario_b", "lag_reduction"),
    ("scenario_b", "risk_premium_reduction_bps"),
    ("scenario_b", "bluewashing_risk_reduction"),
    ("scenario_b", "implementation_cost_pct"),
    ("scenario_b", "legal_uncertainty_penalty"),
    ("scenario_b", "oracle_failure_risk"),
    ("scenario_b", "adoption_friction"),
)


TRIANGULAR_CLI_OVERRIDES = (
    ("mrv_reduction_min", "scenario_b", "mrv_reduction", "min"),
    ("mrv_reduction_max", "scenario_b", "mrv_reduction", "max"),
    ("lag_reduction_min", "scenario_b", "lag_reduction", "min"),
    ("lag_reduction_max", "scenario_b", "lag_reduction", "max"),
    ("premium_reduction_min", "scenario_b", "risk_premium_reduction_bps", "min"),
    ("premium_reduction_max", "scenario_b", "risk_premium_reduction_bps", "max"),
    ("implementation_cost_min", "scenario_b", "implementation_cost_pct", "min"),
    ("implementation_cost_mode", "scenario_b", "implementation_cost_pct", "mode"),
    ("implementation_cost_max", "scenario_b", "implementation_cost_pct", "max"),
    ("legal_uncertainty_min", "scenario_b", "legal_uncertainty_penalty", "min"),
    ("legal_uncertainty_mode", "scenario_b", "legal_uncertainty_penalty", "mode"),
    ("legal_uncertainty_max", "scenario_b", "legal_uncertainty_penalty", "max"),
    ("oracle_risk_min", "scenario_b", "oracle_failure_risk", "min"),
    ("oracle_risk_mode", "scenario_b", "oracle_failure_risk", "mode"),
    ("oracle_risk_max", "scenario_b", "oracle_failure_risk", "max"),
    ("adoption_friction_min", "scenario_b", "adoption_friction", "min"),
    ("adoption_friction_mode", "scenario_b", "adoption_friction", "mode"),
    ("adoption_friction_max", "scenario_b", "adoption_friction", "max"),
)


def load_yaml_config(path):
    """Load configuration from YAML if it exists."""
    config_path = Path(path)
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config at {config_path} must contain a mapping.")

    return loaded


def deep_merge(base, override):
    """Recursively merge two dictionaries without mutating the inputs."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_cli_overrides(config, args):
    """Apply supported CLI overrides to the merged configuration."""
    updated = copy.deepcopy(config)

    if getattr(args, "iterations", None) is not None:
        updated["simulation"]["iterations"] = args.iterations
    if getattr(args, "project_value", None) is not None:
        updated["simulation"]["project_value_euro"] = args.project_value
    if getattr(args, "seed", None) is not None:
        updated["simulation"]["random_seed"] = args.seed
    if getattr(args, "baseline_mobilization", None) is not None:
        updated["mobilization_model"][
            "literature_baseline"
        ] = args.baseline_mobilization

    for attr_name, section, field, point in TRIANGULAR_CLI_OVERRIDES:
        attr_value = getattr(args, attr_name, None)
        if attr_value is not None:
            updated[section][field][point] = attr_value

    return updated


def _validate_triangular(name, values):
    minimum = values["min"]
    mode = values["mode"]
    maximum = values["max"]
    if minimum > mode or mode > maximum:
        raise ValueError(
            f"Invalid triangular distribution for {name}: "
            f"expected min <= mode <= max, got {minimum}, {mode}, {maximum}."
        )


def _validate_bounds(name, values, minimum, maximum=None):
    for point_name, point_value in values.items():
        if point_value < minimum:
            raise ValueError(f"{name}.{point_name} must be >= {minimum}.")
        if maximum is not None and point_value > maximum:
            raise ValueError(f"{name}.{point_name} must be <= {maximum}.")


def validate_config(config):
    """Validate core configuration rules."""
    for section, field in TRIANGULAR_PATHS:
        _validate_triangular(f"{section}.{field}", config[section][field])

    _validate_bounds("scenario_b.implementation_cost_pct", config["scenario_b"]["implementation_cost_pct"], 0.0)
    _validate_bounds("scenario_b.legal_uncertainty_penalty", config["scenario_b"]["legal_uncertainty_penalty"], 0.0, 1.0)
    _validate_bounds("scenario_b.oracle_failure_risk", config["scenario_b"]["oracle_failure_risk"], 0.0, 1.0)
    _validate_bounds("scenario_b.adoption_friction", config["scenario_b"]["adoption_friction"], 0.0, 1.0)

    if config["simulation"]["iterations"] <= 0:
        raise ValueError("simulation.iterations must be greater than zero.")

    if config["simulation"]["project_value_euro"] <= 0:
        raise ValueError("simulation.project_value_euro must be greater than zero.")


def load_config(args):
    """Load defaults, YAML config, and CLI overrides in priority order."""
    default_config_path = Path(__file__).resolve().parent / "params.yaml"
    config_path = (
        Path(args.config) if getattr(args, "config", None) else default_config_path
    )

    yaml_config = load_yaml_config(config_path)
    merged = deep_merge(DEFAULT_CONFIG, yaml_config)
    merged = apply_cli_overrides(merged, args)
    validate_config(merged)
    return merged
