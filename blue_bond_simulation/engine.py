from __future__ import annotations

import numpy as np
import pandas as pd


class BlueBondEngine:
    """Monte Carlo engine for comparing blue bond governance scenarios."""

    def __init__(self, config):
        self.config = config
        self.iterations = int(config["simulation"]["iterations"])
        self.project_value_euro = float(config["simulation"]["project_value_euro"])
        self.rng = np.random.default_rng(config["simulation"]["random_seed"])

    def run(self) -> pd.DataFrame:
        """Run the simulation and return a long-format results DataFrame."""
        scenario_a_df = self._run_scenario_a()
        scenario_b_df = self._run_scenario_b(scenario_a_df)
        return pd.concat([scenario_a_df, scenario_b_df], ignore_index=True)

    def _run_scenario_a(self) -> pd.DataFrame:
        scenario_a = self.config["scenario_a"]

        monitoring_cost_pct = self._draw_triangular(
            scenario_a["monitoring_cost_pct"],
            self.iterations,
        )
        monitoring_cost_euro = self.project_value_euro * monitoring_cost_pct
        disbursement_lag_months = self._draw_triangular(
            scenario_a["disbursement_lag_months"],
            self.iterations,
        )
        mobilization_ratio = self._draw_triangular(
            scenario_a["mobilization_ratio"],
            self.iterations,
        )
        risk_premium_bps = self._draw_triangular(
            scenario_a["risk_premium_bps"],
            self.iterations,
        )
        zeros = np.zeros(self.iterations)

        return pd.DataFrame(
            {
                "iteration": np.arange(1, self.iterations + 1),
                "scenario": scenario_a["name"],
                "monitoring_cost_pct": monitoring_cost_pct,
                "monitoring_cost_euro": monitoring_cost_euro,
                "implementation_cost_pct": zeros,
                "implementation_cost_euro": zeros,
                "total_governance_cost_pct": monitoring_cost_pct,
                "total_governance_cost_euro": monitoring_cost_euro,
                "disbursement_lag_months": disbursement_lag_months,
                "mobilization_ratio": mobilization_ratio,
                "gross_mobilization_ratio": mobilization_ratio,
                "net_mobilization_ratio": mobilization_ratio,
                "risk_premium_bps": risk_premium_bps,
                "mrv_reduction": zeros,
                "lag_reduction": zeros,
                "effective_lag_reduction": zeros,
                "risk_premium_reduction_bps": zeros,
                "effective_risk_premium_reduction_bps": zeros,
                "bluewashing_risk_reduction": zeros,
                "effective_bluewashing_reduction": zeros,
                "legal_uncertainty_penalty": zeros,
                "oracle_failure_risk": zeros,
                "adoption_friction": zeros,
                "gross_benefit": zeros,
                "penalty_factor": zeros,
                "net_benefit": zeros,
            }
        )

    def _run_scenario_b(self, scenario_a_df) -> pd.DataFrame:
        scenario_b = self.config["scenario_b"]
        mobilization_model = self.config["mobilization_model"]

        mrv_reduction = self._draw_triangular(
            scenario_b["mrv_reduction"],
            self.iterations,
        )
        lag_reduction = self._draw_triangular(
            scenario_b["lag_reduction"],
            self.iterations,
        )
        risk_premium_reduction_bps = self._draw_triangular(
            scenario_b["risk_premium_reduction_bps"],
            self.iterations,
        )
        bluewashing_risk_reduction = self._draw_triangular(
            scenario_b["bluewashing_risk_reduction"],
            self.iterations,
        )
        implementation_cost_pct = self._draw_triangular(
            scenario_b["implementation_cost_pct"],
            self.iterations,
        )
        legal_uncertainty_penalty = self._draw_triangular(
            scenario_b["legal_uncertainty_penalty"],
            self.iterations,
        )
        oracle_failure_risk = self._draw_triangular(
            scenario_b["oracle_failure_risk"],
            self.iterations,
        )
        adoption_friction = self._draw_triangular(
            scenario_b["adoption_friction"],
            self.iterations,
        )

        effective_lag_reduction = lag_reduction * (1.0 - adoption_friction)
        effective_bluewashing_reduction = bluewashing_risk_reduction * (
            1.0 - oracle_failure_risk
        )
        monitoring_cost_pct = scenario_a_df["monitoring_cost_pct"].to_numpy() * (
            1.0 - mrv_reduction
        )
        monitoring_cost_euro = self.project_value_euro * monitoring_cost_pct
        implementation_cost_euro = self.project_value_euro * implementation_cost_pct
        total_governance_cost_pct = monitoring_cost_pct + implementation_cost_pct
        total_governance_cost_euro = monitoring_cost_euro + implementation_cost_euro
        disbursement_lag_months = scenario_a_df["disbursement_lag_months"].to_numpy() * (
            1.0 - effective_lag_reduction
        )

        premium_adjustment_factor = np.maximum(
            1.0
            - legal_uncertainty_penalty
            - oracle_failure_risk
            - adoption_friction,
            0.0,
        )
        effective_risk_premium_reduction_bps = (
            risk_premium_reduction_bps * premium_adjustment_factor
        )
        risk_premium_bps = np.maximum(
            scenario_a_df["risk_premium_bps"].to_numpy()
            - effective_risk_premium_reduction_bps,
            0.0,
        )

        gross_benefit = (
            mobilization_model["mrv_weight"] * mrv_reduction
            + mobilization_model["lag_weight"] * effective_lag_reduction
            + mobilization_model["premium_weight"]
            * (risk_premium_reduction_bps / 100.0)
            + mobilization_model["bluewashing_weight"]
            * effective_bluewashing_reduction
        )
        penalty_factor = (
            legal_uncertainty_penalty + oracle_failure_risk + adoption_friction
        )
        net_benefit = np.maximum(gross_benefit - penalty_factor, -0.50)

        baseline_mobilization = scenario_a_df["mobilization_ratio"].to_numpy()
        gross_mobilization_ratio = np.minimum(
            baseline_mobilization * (1.0 + gross_benefit),
            mobilization_model["max_mobilization_ratio"],
        )
        net_mobilization_ratio = np.minimum(
            baseline_mobilization * (1.0 + net_benefit),
            mobilization_model["max_mobilization_ratio"],
        )

        return pd.DataFrame(
            {
                "iteration": scenario_a_df["iteration"].to_numpy(),
                "scenario": scenario_b["name"],
                "monitoring_cost_pct": monitoring_cost_pct,
                "monitoring_cost_euro": monitoring_cost_euro,
                "implementation_cost_pct": implementation_cost_pct,
                "implementation_cost_euro": implementation_cost_euro,
                "total_governance_cost_pct": total_governance_cost_pct,
                "total_governance_cost_euro": total_governance_cost_euro,
                "disbursement_lag_months": disbursement_lag_months,
                "mobilization_ratio": net_mobilization_ratio,
                "gross_mobilization_ratio": gross_mobilization_ratio,
                "net_mobilization_ratio": net_mobilization_ratio,
                "risk_premium_bps": risk_premium_bps,
                "mrv_reduction": mrv_reduction,
                "lag_reduction": lag_reduction,
                "effective_lag_reduction": effective_lag_reduction,
                "risk_premium_reduction_bps": risk_premium_reduction_bps,
                "effective_risk_premium_reduction_bps": effective_risk_premium_reduction_bps,
                "bluewashing_risk_reduction": bluewashing_risk_reduction,
                "effective_bluewashing_reduction": effective_bluewashing_reduction,
                "legal_uncertainty_penalty": legal_uncertainty_penalty,
                "oracle_failure_risk": oracle_failure_risk,
                "adoption_friction": adoption_friction,
                "gross_benefit": gross_benefit,
                "penalty_factor": penalty_factor,
                "net_benefit": net_benefit,
            }
        )

    def summarize(self, results_df) -> pd.DataFrame:
        """Summarize the main model metrics for each scenario."""
        metrics = [
            "mobilization_ratio",
            "gross_mobilization_ratio",
            "net_mobilization_ratio",
            "monitoring_cost_pct",
            "monitoring_cost_euro",
            "implementation_cost_pct",
            "implementation_cost_euro",
            "total_governance_cost_pct",
            "total_governance_cost_euro",
            "disbursement_lag_months",
            "risk_premium_bps",
            "gross_benefit",
            "penalty_factor",
            "net_benefit",
            "effective_risk_premium_reduction_bps",
        ]
        rows = []

        for scenario_name, scenario_df in results_df.groupby("scenario"):
            for metric in metrics:
                values = scenario_df[metric].to_numpy()
                rows.append(
                    {
                        "scenario": scenario_name,
                        "metric": metric,
                        "mean": np.mean(values),
                        "median": np.median(values),
                        "p10": np.percentile(values, 10),
                        "p25": np.percentile(values, 25),
                        "p75": np.percentile(values, 75),
                        "p90": np.percentile(values, 90),
                        "standard_deviation": np.std(values, ddof=1),
                    }
                )

        return pd.DataFrame(rows)

    def calculate_probability_b_beats_a(self, results_df) -> dict:
        """Calculate the probability that Scenario B outperforms Scenario A."""
        scenario_a_name = self.config["scenario_a"]["name"]
        scenario_b_name = self.config["scenario_b"]["name"]

        scenario_a_df = (
            results_df[results_df["scenario"] == scenario_a_name]
            .sort_values("iteration")
            .set_index("iteration")
        )
        scenario_b_df = (
            results_df[results_df["scenario"] == scenario_b_name]
            .sort_values("iteration")
            .set_index("iteration")
        )

        return {
            "mobilization_ratio": float(
                (
                    scenario_b_df["mobilization_ratio"]
                    > scenario_a_df["mobilization_ratio"]
                ).mean()
            ),
            "monitoring_cost": float(
                (
                    scenario_b_df["total_governance_cost_euro"]
                    < scenario_a_df["total_governance_cost_euro"]
                ).mean()
            ),
            "governance_cost": float(
                (
                    scenario_b_df["total_governance_cost_euro"]
                    < scenario_a_df["total_governance_cost_euro"]
                ).mean()
            ),
            "disbursement_lag": float(
                (
                    scenario_b_df["disbursement_lag_months"]
                    < scenario_a_df["disbursement_lag_months"]
                ).mean()
            ),
            "risk_premium": float(
                (
                    scenario_b_df["risk_premium_bps"]
                    < scenario_a_df["risk_premium_bps"]
                ).mean()
            ),
        }

    def build_tornado_data(self, results_df) -> pd.DataFrame:
        """Prepare one-at-a-time sensitivity results for Scenario B mobilization."""
        scenario_a_name = self.config["scenario_a"]["name"]
        scenario_b_name = self.config["scenario_b"]["name"]

        scenario_a_df = results_df[results_df["scenario"] == scenario_a_name]
        scenario_b_df = results_df[results_df["scenario"] == scenario_b_name]

        baseline_values = self._percentiles(
            scenario_a_df["mobilization_ratio"].to_numpy()
        )
        mrv_values = self._percentiles(scenario_b_df["mrv_reduction"].to_numpy())
        lag_values = self._percentiles(scenario_b_df["lag_reduction"].to_numpy())
        premium_values = self._percentiles(
            scenario_b_df["risk_premium_reduction_bps"].to_numpy()
        )
        bluewashing_values = self._percentiles(
            scenario_b_df["bluewashing_risk_reduction"].to_numpy()
        )
        legal_values = self._percentiles(
            scenario_b_df["legal_uncertainty_penalty"].to_numpy()
        )
        oracle_values = self._percentiles(
            scenario_b_df["oracle_failure_risk"].to_numpy()
        )
        adoption_values = self._percentiles(
            scenario_b_df["adoption_friction"].to_numpy()
        )

        base_output = self._scenario_b_net_mobilization_ratio(
            baseline_mobilization=baseline_values["p50"],
            mrv_reduction=mrv_values["p50"],
            lag_reduction=lag_values["p50"],
            risk_premium_reduction_bps=premium_values["p50"],
            bluewashing_risk_reduction=bluewashing_values["p50"],
            legal_uncertainty_penalty=legal_values["p50"],
            oracle_failure_risk=oracle_values["p50"],
            adoption_friction=adoption_values["p50"],
        )

        parameter_specs = [
            ("Baseline mobilization", baseline_values, "baseline"),
            ("MRV cost reduction", mrv_values, "mrv"),
            ("Disbursement speed improvement", lag_values, "lag"),
            ("Risk premium reduction", premium_values, "premium"),
            ("Blue-washing risk reduction", bluewashing_values, "bluewashing"),
            ("Legal uncertainty", legal_values, "legal"),
            ("Oracle failure risk", oracle_values, "oracle"),
            ("Adoption friction", adoption_values, "adoption"),
        ]

        rows = []
        for label, percentiles, parameter_name in parameter_specs:
            low_output = self._tornado_output(
                parameter_name=parameter_name,
                parameter_value=percentiles["p10"],
                baseline_mobilization=baseline_values["p50"],
                mrv_reduction=mrv_values["p50"],
                lag_reduction=lag_values["p50"],
                premium_reduction=premium_values["p50"],
                bluewashing_reduction=bluewashing_values["p50"],
                legal_uncertainty_penalty=legal_values["p50"],
                oracle_failure_risk=oracle_values["p50"],
                adoption_friction=adoption_values["p50"],
            )
            high_output = self._tornado_output(
                parameter_name=parameter_name,
                parameter_value=percentiles["p90"],
                baseline_mobilization=baseline_values["p50"],
                mrv_reduction=mrv_values["p50"],
                lag_reduction=lag_values["p50"],
                premium_reduction=premium_values["p50"],
                bluewashing_reduction=bluewashing_values["p50"],
                legal_uncertainty_penalty=legal_values["p50"],
                oracle_failure_risk=oracle_values["p50"],
                adoption_friction=adoption_values["p50"],
            )
            rows.append(
                {
                    "parameter": label,
                    "base_output": base_output,
                    "low_output": low_output,
                    "high_output": high_output,
                    "impact": abs(high_output - low_output),
                }
            )

        tornado_df = pd.DataFrame(rows).sort_values("impact", ascending=False)
        return tornado_df.reset_index(drop=True)

    def _draw_triangular(self, distribution, size):
        return self.rng.triangular(
            distribution["min"],
            distribution["mode"],
            distribution["max"],
            size=size,
        )

    def _percentiles(self, values):
        return {
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        }

    def _scenario_b_net_mobilization_ratio(
        self,
        baseline_mobilization,
        mrv_reduction,
        lag_reduction,
        risk_premium_reduction_bps,
        bluewashing_risk_reduction,
        legal_uncertainty_penalty,
        oracle_failure_risk,
        adoption_friction,
    ):
        model = self.config["mobilization_model"]
        effective_lag_reduction = lag_reduction * (1.0 - adoption_friction)
        effective_bluewashing_reduction = bluewashing_risk_reduction * (
            1.0 - oracle_failure_risk
        )
        gross_benefit = (
            model["mrv_weight"] * mrv_reduction
            + model["lag_weight"] * effective_lag_reduction
            + model["premium_weight"] * (risk_premium_reduction_bps / 100.0)
            + model["bluewashing_weight"] * effective_bluewashing_reduction
        )
        penalty_factor = (
            legal_uncertainty_penalty + oracle_failure_risk + adoption_friction
        )
        net_benefit = max(gross_benefit - penalty_factor, -0.50)
        mobilization_ratio = baseline_mobilization * (1.0 + net_benefit)
        return min(mobilization_ratio, model["max_mobilization_ratio"])

    def _tornado_output(
        self,
        parameter_name,
        parameter_value,
        baseline_mobilization,
        mrv_reduction,
        lag_reduction,
        premium_reduction,
        bluewashing_reduction,
        legal_uncertainty_penalty,
        oracle_failure_risk,
        adoption_friction,
    ):
        if parameter_name == "baseline":
            baseline_mobilization = parameter_value
        elif parameter_name == "mrv":
            mrv_reduction = parameter_value
        elif parameter_name == "lag":
            lag_reduction = parameter_value
        elif parameter_name == "premium":
            premium_reduction = parameter_value
        elif parameter_name == "bluewashing":
            bluewashing_reduction = parameter_value
        elif parameter_name == "legal":
            legal_uncertainty_penalty = parameter_value
        elif parameter_name == "oracle":
            oracle_failure_risk = parameter_value
        elif parameter_name == "adoption":
            adoption_friction = parameter_value

        return self._scenario_b_net_mobilization_ratio(
            baseline_mobilization=baseline_mobilization,
            mrv_reduction=mrv_reduction,
            lag_reduction=lag_reduction,
            risk_premium_reduction_bps=premium_reduction,
            bluewashing_risk_reduction=bluewashing_reduction,
            legal_uncertainty_penalty=legal_uncertainty_penalty,
            oracle_failure_risk=oracle_failure_risk,
            adoption_friction=adoption_friction,
        )
