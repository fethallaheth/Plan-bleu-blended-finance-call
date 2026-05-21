from __future__ import annotations

import argparse
from pathlib import Path

from ..charts import generate_all_charts
from ..config import load_config
from ..engine import BlueBondEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation for Mediterranean blended blue bonds."
    )
    parser.add_argument("--config", help="Path to a YAML configuration file.")
    parser.add_argument("--iterations", type=int, help="Number of Monte Carlo iterations.")
    parser.add_argument(
        "--project-value",
        dest="project_value",
        type=float,
        help="Project value in euros.",
    )
    parser.add_argument(
        "--mrv-reduction-min",
        dest="mrv_reduction_min",
        type=float,
        help="Minimum MRV cost reduction for Scenario B.",
    )
    parser.add_argument(
        "--mrv-reduction-max",
        dest="mrv_reduction_max",
        type=float,
        help="Maximum MRV cost reduction for Scenario B.",
    )
    parser.add_argument(
        "--lag-reduction-min",
        dest="lag_reduction_min",
        type=float,
        help="Minimum disbursement lag reduction for Scenario B.",
    )
    parser.add_argument(
        "--lag-reduction-max",
        dest="lag_reduction_max",
        type=float,
        help="Maximum disbursement lag reduction for Scenario B.",
    )
    parser.add_argument(
        "--premium-reduction-min",
        dest="premium_reduction_min",
        type=float,
        help="Minimum risk premium reduction in basis points for Scenario B.",
    )
    parser.add_argument(
        "--premium-reduction-max",
        dest="premium_reduction_max",
        type=float,
        help="Maximum risk premium reduction in basis points for Scenario B.",
    )
    parser.add_argument(
        "--implementation-cost-min",
        dest="implementation_cost_min",
        type=float,
        help="Minimum implementation cost percentage for Scenario B.",
    )
    parser.add_argument(
        "--implementation-cost-mode",
        dest="implementation_cost_mode",
        type=float,
        help="Mode implementation cost percentage for Scenario B.",
    )
    parser.add_argument(
        "--implementation-cost-max",
        dest="implementation_cost_max",
        type=float,
        help="Maximum implementation cost percentage for Scenario B.",
    )
    parser.add_argument(
        "--legal-uncertainty-min",
        dest="legal_uncertainty_min",
        type=float,
        help="Minimum legal uncertainty penalty for Scenario B.",
    )
    parser.add_argument(
        "--legal-uncertainty-mode",
        dest="legal_uncertainty_mode",
        type=float,
        help="Mode legal uncertainty penalty for Scenario B.",
    )
    parser.add_argument(
        "--legal-uncertainty-max",
        dest="legal_uncertainty_max",
        type=float,
        help="Maximum legal uncertainty penalty for Scenario B.",
    )
    parser.add_argument(
        "--oracle-risk-min",
        dest="oracle_risk_min",
        type=float,
        help="Minimum oracle failure risk for Scenario B.",
    )
    parser.add_argument(
        "--oracle-risk-mode",
        dest="oracle_risk_mode",
        type=float,
        help="Mode oracle failure risk for Scenario B.",
    )
    parser.add_argument(
        "--oracle-risk-max",
        dest="oracle_risk_max",
        type=float,
        help="Maximum oracle failure risk for Scenario B.",
    )
    parser.add_argument(
        "--adoption-friction-min",
        dest="adoption_friction_min",
        type=float,
        help="Minimum adoption friction for Scenario B.",
    )
    parser.add_argument(
        "--adoption-friction-mode",
        dest="adoption_friction_mode",
        type=float,
        help="Mode adoption friction for Scenario B.",
    )
    parser.add_argument(
        "--adoption-friction-max",
        dest="adoption_friction_max",
        type=float,
        help="Maximum adoption friction for Scenario B.",
    )
    parser.add_argument(
        "--baseline-mobilization",
        dest="baseline_mobilization",
        type=float,
        help="Override the literature baseline mobilization ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def print_summary(summary_df, config):
    ordered_scenarios = [config["scenario_a"]["name"], config["scenario_b"]["name"]]

    print("\nSummary Statistics")
    print("=" * 80)
    for scenario_name in ordered_scenarios:
        scenario_summary = summary_df[summary_df["scenario"] == scenario_name].copy()
        scenario_summary = scenario_summary.set_index("metric")[
            [
                "mean",
                "median",
                "p10",
                "p25",
                "p75",
                "p90",
                "standard_deviation",
            ]
        ]
        print(f"\n{scenario_name}")
        print(scenario_summary.round(4).to_string())


def print_probabilities(probabilities):
    print("\nProbability Scenario B Beats Scenario A")
    print("=" * 80)
    print(f"Mobilization ratio: {probabilities['mobilization_ratio']:.2%}")
    print(f"Governance cost: {probabilities['governance_cost']:.2%}")
    print(f"Disbursement lag: {probabilities['disbursement_lag']:.2%}")
    print(f"Risk premium: {probabilities['risk_premium']:.2%}")


def print_penalty_diagnostics(results_df, config):
    scenario_b_name = config["scenario_b"]["name"]
    scenario_b_df = results_df[results_df["scenario"] == scenario_b_name]
    metrics = [
        "implementation_cost_pct",
        "legal_uncertainty_penalty",
        "oracle_failure_risk",
        "adoption_friction",
        "gross_benefit",
        "penalty_factor",
        "net_benefit",
        "total_governance_cost_pct",
        "effective_risk_premium_reduction_bps",
    ]

    print("\nScenario B Penalty Diagnostics")
    print("=" * 80)
    for metric in metrics:
        median_value = float(scenario_b_df[metric].median())
        print(f"{metric}: {median_value:.4f}")


def main():
    args = parse_args()
    config = load_config(args)

    engine = BlueBondEngine(config)
    results_df = engine.run()
    summary_df = engine.summarize(results_df)
    probabilities = engine.calculate_probability_b_beats_a(results_df)
    tornado_df = engine.build_tornado_data(results_df)

    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / config["charting"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "simulation_results.csv", index=False)
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)
    generate_all_charts(results_df, tornado_df, config, output_dir)

    print_summary(summary_df, config)
    print_probabilities(probabilities)
    print_penalty_diagnostics(results_df, config)
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
