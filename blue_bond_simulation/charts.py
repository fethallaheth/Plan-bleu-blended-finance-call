from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PALETTE = ["#355C7D", "#6C9A8B"]
STACK_PALETTE = {"monitoring": "#355C7D", "implementation": "#E09F3E"}


def generate_all_charts(results_df, tornado_df, config, output_dir):
    """Generate all required charts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")

    create_catalytic_efficiency_chart(results_df, config, output_path)
    create_risk_premium_chart(results_df, config, output_path)
    create_monitoring_cost_chart(results_df, config, output_path)
    create_tornado_chart(tornado_df, config, output_path)
    create_cdf_overlay_chart(results_df, config, output_path)


def create_catalytic_efficiency_chart(results_df, config, output_dir):
    order = _scenario_order(config)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(
        data=results_df,
        x="mobilization_ratio",
        y="scenario",
        order=order,
        whis=(10, 90),
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        fliersize=0,
        linewidth=1.2,
        palette=PALETTE,
        ax=ax,
    )

    baseline = config["mobilization_model"]["literature_baseline"]
    ax.axvline(baseline, color="dimgray", linestyle="--", linewidth=1.2)
    ax.text(
        baseline + 0.02,
        0.98,
        f"Literature baseline = {baseline:.1f}",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=9,
        color="dimgray",
    )
    ax.set_title("Catalytic Efficiency Distribution")
    ax.set_xlabel("Mobilization ratio")
    ax.set_ylabel("Scenario")
    fig.tight_layout()
    _save_figure(fig, output_dir / "chart01_catalytic_efficiency.png", config)


def create_risk_premium_chart(results_df, config, output_dir):
    order = _scenario_order(config)
    stats = _metric_stats(results_df, "risk_premium_bps", order)

    fig, ax = plt.subplots(figsize=(8, 6))
    positions = np.arange(len(order))
    medians = stats["median"].to_numpy()
    lower_errors = medians - stats["p10"].to_numpy()
    upper_errors = stats["p90"].to_numpy() - medians

    ax.bar(positions, medians, color=PALETTE, width=0.6)
    ax.errorbar(
        positions,
        medians,
        yerr=[lower_errors, upper_errors],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=6,
    )

    reduction = medians[0] - medians[1]
    bracket_y = stats["p90"].max() + 18
    ax.plot(
        [positions[0], positions[0], positions[1], positions[1]],
        [bracket_y - 6, bracket_y, bracket_y, bracket_y - 6],
        color="black",
        linewidth=1.2,
    )
    ax.text(
        positions.mean(),
        bracket_y + 5,
        f"Median reduction: {reduction:.1f} bps",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.set_xticks(positions, order)
    ax.set_ylabel("Risk premium (basis points)")
    ax.set_title("Risk Premium After Risk Adjustment")
    fig.tight_layout()
    _save_figure(fig, output_dir / "chart02_risk_premium.png", config)


def create_monitoring_cost_chart(results_df, config, output_dir):
    order = _scenario_order(config)
    monitoring_stats = _metric_stats(results_df, "monitoring_cost_pct", order).copy()
    implementation_stats = _metric_stats(
        results_df, "implementation_cost_pct", order
    ).copy()
    total_stats = _metric_stats(results_df, "total_governance_cost_pct", order).copy()

    for stats in (monitoring_stats, implementation_stats, total_stats):
        for column in ("median", "p25", "p75"):
            stats[column] = stats[column] * 100.0

    fig, ax = plt.subplots(figsize=(8, 6))
    positions = np.arange(len(order))
    monitoring_medians = monitoring_stats["median"].to_numpy()
    implementation_medians = implementation_stats["median"].to_numpy()
    total_medians = total_stats["median"].to_numpy()

    ax.bar(
        positions,
        monitoring_medians,
        color=STACK_PALETTE["monitoring"],
        width=0.6,
        label="Monitoring cost",
    )
    ax.bar(
        positions,
        implementation_medians,
        bottom=monitoring_medians,
        color=STACK_PALETTE["implementation"],
        width=0.6,
        label="Implementation cost",
    )

    for x_pos, total_median, p25, p75 in zip(
        positions,
        total_medians,
        total_stats["p25"].to_numpy(),
        total_stats["p75"].to_numpy(),
    ):
        ax.vlines(x_pos, p25, p75, color="black", linewidth=1.2)
        ax.hlines([p25, p75], x_pos - 0.08, x_pos + 0.08, color="black", linewidth=1.2)
        ax.text(
            x_pos,
            total_median + 0.18,
            f"{total_median:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.text(
        0.02,
        0.98,
        config["charting"]["artemis_note"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    ax.set_xticks(positions, order)
    ax.set_ylabel("Governance cost (% of project value)")
    ax.set_title("Monitoring and Governance Cost")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir / "chart03_monitoring_costs.png", config)


def create_tornado_chart(tornado_df, config, output_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = tornado_df["parameter"].to_list()
    low_output = tornado_df["low_output"].to_numpy()
    base_output = tornado_df["base_output"].to_numpy()
    high_output = tornado_df["high_output"].to_numpy()
    positions = np.arange(len(labels))

    ax.barh(
        positions,
        base_output - low_output,
        left=low_output,
        color="#B56576",
        alpha=0.9,
        label="P10 to median",
    )
    ax.barh(
        positions,
        high_output - base_output,
        left=base_output,
        color="#5B8E7D",
        alpha=0.9,
        label="Median to P90",
    )

    base_line = float(tornado_df["base_output"].iloc[0])
    ax.axvline(base_line, color="black", linestyle="--", linewidth=1.1)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Scenario B net mobilization ratio")
    ax.set_title("Tornado Sensitivity for Catalytic Efficiency")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir / "chart04_tornado_sensitivity.png", config)


def create_cdf_overlay_chart(results_df, config, output_dir):
    order = _scenario_order(config)
    scenario_a_values = np.sort(
        results_df.loc[
            results_df["scenario"] == order[0],
            "mobilization_ratio",
        ].to_numpy()
    )
    scenario_b_values = np.sort(
        results_df.loc[
            results_df["scenario"] == order[1],
            "mobilization_ratio",
        ].to_numpy()
    )

    scenario_a_cdf = np.arange(1, len(scenario_a_values) + 1) / len(scenario_a_values)
    scenario_b_cdf = np.arange(1, len(scenario_b_values) + 1) / len(scenario_b_values)

    x_min = float(min(scenario_a_values.min(), scenario_b_values.min()))
    x_max = float(max(scenario_a_values.max(), scenario_b_values.max()))
    common_x = np.linspace(x_min, x_max, 500)
    interp_a = np.interp(
        common_x, scenario_a_values, scenario_a_cdf, left=0.0, right=1.0
    )
    interp_b = np.interp(
        common_x, scenario_b_values, scenario_b_cdf, left=0.0, right=1.0
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        common_x,
        interp_a,
        linestyle="--",
        linewidth=2,
        color=PALETTE[0],
        label=order[0],
    )
    ax.plot(
        common_x,
        interp_b,
        linestyle="-",
        linewidth=2,
        color=PALETTE[1],
        label=order[1],
    )
    ax.fill_between(common_x, interp_a, interp_b, color=PALETTE[1], alpha=0.35)

    percentile_values = {
        "P25": np.percentile(scenario_b_values, 25),
        "P50": np.percentile(scenario_b_values, 50),
        "P75": np.percentile(scenario_b_values, 75),
    }
    label_y_positions = {"P25": 0.22, "P50": 0.52, "P75": 0.82}
    for label, value in percentile_values.items():
        ax.axvline(value, color="gray", linestyle=":", linewidth=1.1)
        ax.text(value + 0.02, label_y_positions[label], label, color="gray", fontsize=9)

    ax.set_xlabel("Mobilization ratio")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("CDF of Catalytic Efficiency")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir / "chart05_cdf_overlay.png", config)


def _metric_stats(results_df, metric, order):
    stats = []
    for scenario_name in order:
        values = results_df.loc[results_df["scenario"] == scenario_name, metric].to_numpy()
        stats.append(
            {
                "scenario": scenario_name,
                "median": np.median(values),
                "p10": np.percentile(values, 10),
                "p25": np.percentile(values, 25),
                "p75": np.percentile(values, 75),
                "p90": np.percentile(values, 90),
            }
        )
    return pd.DataFrame(stats)


def _scenario_order(config):
    return [config["scenario_a"]["name"], config["scenario_b"]["name"]]


def _save_figure(fig, output_path, config):
    fig.savefig(
        output_path,
        dpi=config["charting"]["figure_dpi"],
        format=config["charting"]["figure_format"],
        bbox_inches="tight",
    )
    plt.close(fig)
