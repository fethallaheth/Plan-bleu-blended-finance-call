from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .config import ARTEMIS_ANCHOR, CHART_STYLE, FRANCE_STRESS_SCENARIOS, MODEL_PARAMETERS, sigmoid
except ImportError:
    from config import ARTEMIS_ANCHOR, CHART_STYLE, FRANCE_STRESS_SCENARIOS, MODEL_PARAMETERS, sigmoid


def generate_france_figures(
    pathway_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    iterations_df: pd.DataFrame,
    pathway_integrity_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    _chart_pathway_radar(pathway_df, output_dir)
    _chart_pathway_bar(pathway_df, output_dir)
    _chart_summary_table(summary_df, output_dir)
    _chart_mobilization_boxplot(iterations_df, output_dir)
    _chart_issuance_collapse(summary_df, output_dir)
    _chart_risk_premium(summary_df, output_dir)
    _chart_governance_cost(summary_df, output_dir)
    _chart_pes(summary_df, output_dir)
    _chart_data_admissibility(summary_df, output_dir)
    _chart_time_to_issuance(summary_df, output_dir)
    _chart_pathway_integrity_heatmap(pathway_integrity_df, output_dir)
    _chart_dashboard(summary_df, iterations_df, output_dir)
    _chart_cliff_curve(summary_df, output_dir)
    _chart_waterfall(summary_df, output_dir)
    _chart_severity_map(summary_df, output_dir)


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    dpi = int(CHART_STYLE["dpi"])
    for extension in CHART_STYLE["formats"]:
        fig.savefig(output_dir / f"{name}.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _ordered_labels() -> list[str]:
    return [payload["label"] for payload in FRANCE_STRESS_SCENARIOS.values()]


def _chart_pathway_radar(pathway_df: pd.DataFrame, output_dir: Path) -> None:
    labels = pathway_df["pathway"].to_list()
    values = pathway_df["score"].to_numpy()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values = np.concatenate([values, values[:1]])
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.plot(angles, values, color="#1f3b73", linewidth=2)
    ax.fill(angles, values, color="#1f3b73", alpha=0.25)
    ax.plot(angles, np.full_like(angles, 70.0), color="crimson", linestyle="--", linewidth=1.2)
    ax.set_xticks(np.linspace(0, 2 * np.pi, len(labels), endpoint=False))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("France Institutional Pathway Radar", pad=20)
    _save_figure(fig, output_dir, "france_institutional_pathway_radar")


def _chart_pathway_bar(pathway_df: pd.DataFrame, output_dir: Path) -> None:
    chart_df = pathway_df.sort_values("score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(chart_df["pathway"], chart_df["score"], color="#2e6f95")
    ax.axvline(70.0, color="crimson", linestyle="--", linewidth=1.2)
    for bar in bars:
        ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.1f}", va="center")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Pathway Score")
    ax.set_title("France Pathway Scores (Sorted)")
    _save_figure(fig, output_dir, "france_pathway_score_bar")


def _chart_summary_table(summary_df: pd.DataFrame, output_dir: Path) -> None:
    display_df = summary_df[
        [
            "scenario_label",
            "institutional_quality",
            "median_mobilization_ratio",
            "median_risk_premium_bps",
            "median_governance_cost_pct",
            "median_issuance_probability",
            "median_pes_achievement_eur",
            "median_data_admissibility_probability",
            "median_time_to_issuance_months",
            "status",
        ]
    ].copy()
    display_df["institutional_quality"] = (display_df["institutional_quality"] * 100.0).round(1)
    display_df["median_governance_cost_pct"] = (display_df["median_governance_cost_pct"] * 100.0).round(2)
    display_df["median_issuance_probability"] = (display_df["median_issuance_probability"] * 100.0).round(1)
    display_df["median_data_admissibility_probability"] = (
        display_df["median_data_admissibility_probability"] * 100.0
    ).round(1)
    display_df = display_df.round(
        {
            "median_mobilization_ratio": 3,
            "median_risk_premium_bps": 1,
            "median_pes_achievement_eur": 0,
            "median_time_to_issuance_months": 1,
        }
    )

    fig, ax = plt.subplots(figsize=(18, 3))
    ax.axis("off")
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax.set_title("France Stress Scenario Summary", pad=12)
    _save_figure(fig, output_dir, "france_stress_summary_table")


def _chart_mobilization_boxplot(iterations_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=iterations_df,
        x="scenario_label",
        y="mobilization_ratio",
        order=order,
        whis=(10, 90),
        showfliers=False,
        color="#6ca6c1",
        ax=ax,
    )
    ax.axhline(4.0, color="crimson", linestyle="--", linewidth=1.1)
    ax.set_title("Mobilization Ratio Under Stress")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mobilization Ratio")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, output_dir, "stress_mobilization_boxplot")


def _chart_issuance_collapse(summary_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(chart_df["scenario_label"], chart_df["median_issuance_probability"] * 100.0, color="#1f3b73")
    ax.set_ylim(0, 100)
    ax.set_title("Bond Issuance Probability Collapse by Scenario")
    ax.set_ylabel("Issuance Probability (%)")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, output_dir, "stress_issuance_probability_collapse")


def _chart_risk_premium(summary_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    values = chart_df["median_risk_premium_bps"].to_numpy()
    lower = values - chart_df["risk_premium_p10"].to_numpy()
    upper = chart_df["risk_premium_p90"].to_numpy() - values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(chart_df["scenario_label"], values, color="#2e6f95")
    ax.errorbar(chart_df["scenario_label"], values, yerr=[lower, upper], fmt="none", capsize=5, color="black")
    ax.set_title("Risk Premium by Stress Scenario")
    ax.set_ylabel("Risk Premium (bps)")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, output_dir, "stress_risk_premium_by_scenario")


def _chart_governance_cost(summary_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    values = chart_df["median_governance_cost_pct"].to_numpy() * 100.0
    lower = values - chart_df["governance_cost_p10"].to_numpy() * 100.0
    upper = chart_df["governance_cost_p90"].to_numpy() * 100.0 - values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(chart_df["scenario_label"], values, color="#4c956c")
    ax.errorbar(chart_df["scenario_label"], values, yerr=[lower, upper], fmt="none", capsize=5, color="black")
    ax.set_title("Governance Cost by Stress Scenario")
    ax.set_ylabel("Governance Cost (% of project value)")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, output_dir, "stress_governance_cost_by_scenario")


def _chart_pes(summary_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(chart_df["scenario_label"], chart_df["median_pes_achievement_eur"], color="#6ca6c1")
    ax.axhline(2_000_000, color="crimson", linestyle="--", linewidth=1.2)
    ax.set_title("PES Achievement Under Stress")
    ax.set_ylabel("PES Achievement (EUR)")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, output_dir, "stress_pes_achievement")


def _chart_data_admissibility(summary_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    colors = ["#2e6f95"] * len(chart_df)
    highlight = "B: Greenwashing / Oracle Data Challenge"
    if highlight in chart_df["scenario_label"].values:
        colors[chart_df["scenario_label"].to_list().index(highlight)] = "#b22222"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(chart_df["scenario_label"], chart_df["median_data_admissibility_probability"] * 100.0, color=colors)
    ax.set_title("Data Admissibility Probability Under Stress")
    ax.set_ylabel("Data Admissibility (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, output_dir, "stress_data_admissibility")


def _chart_time_to_issuance(summary_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(chart_df["scenario_label"], chart_df["median_time_to_issuance_months"], color="#1f3b73")
    ax.axhline(30.0, color="crimson", linestyle="--", linewidth=1.2)
    ax.text(
        0.02,
        0.96,
        "30-month reference line = MODEL_ASSUMPTION",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.set_title("Time to Issuance Under Stress")
    ax.set_ylabel("Months")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, output_dir, "stress_time_to_issuance")


def _chart_pathway_integrity_heatmap(pathway_integrity_df: pd.DataFrame, output_dir: Path) -> None:
    chart_df = pathway_integrity_df.set_index("pathway")
    rename_map = {key: payload["label"] for key, payload in FRANCE_STRESS_SCENARIOS.items()}
    chart_df = chart_df.rename(columns=rename_map)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(chart_df, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5, cbar_kws={"label": "Integrity Score"}, ax=ax)
    ax.set_title("Pathway Integrity Heatmap by Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("France Pathways")
    _save_figure(fig, output_dir, "pathway_integrity_heatmap")


def _chart_dashboard(summary_df: pd.DataFrame, iterations_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    sns.boxplot(data=iterations_df, x="scenario_label", y="mobilization_ratio", order=order, showfliers=False, ax=axes[0, 0], color="#6ca6c1")
    axes[0, 0].axhline(4.0, color="crimson", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("Mobilization Ratio")
    axes[0, 0].tick_params(axis="x", rotation=20)

    axes[0, 1].bar(chart_df["scenario_label"], chart_df["median_issuance_probability"] * 100.0, color="#1f3b73")
    axes[0, 1].set_title("Issuance Probability (%)")
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].tick_params(axis="x", rotation=20)

    axes[1, 0].bar(chart_df["scenario_label"], chart_df["median_risk_premium_bps"], color="#2e6f95")
    axes[1, 0].set_title("Risk Premium (bps)")
    axes[1, 0].tick_params(axis="x", rotation=20)

    axes[1, 1].bar(chart_df["scenario_label"], chart_df["median_time_to_issuance_months"], color="#4c956c")
    axes[1, 1].axhline(30.0, color="crimson", linestyle="--", linewidth=1.0)
    axes[1, 1].set_title("Time to Issuance (months)")
    axes[1, 1].tick_params(axis="x", rotation=20)

    fig.suptitle("France Stress Dashboard", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_dir, "france_stress_dashboard")


def _chart_cliff_curve(summary_df: pd.DataFrame, output_dir: Path) -> None:
    slope = float(MODEL_PARAMETERS["issuance_sigmoid_scale"]["value"])
    quality_points = np.linspace(0, 100, 500)
    probabilities = sigmoid(((quality_points / 100.0) - 0.50) * slope) * 100.0
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(quality_points, probabilities, color="#1f3b73", linewidth=2)
    ax.axvline(50.0, color="crimson", linestyle="--", linewidth=1.0)
    for _, row in summary_df.iterrows():
        ax.scatter(row["institutional_quality"] * 100.0, row["median_issuance_probability"] * 100.0, color="#b22222")
        ax.text(
            row["institutional_quality"] * 100.0 + 0.8,
            row["median_issuance_probability"] * 100.0,
            row["scenario_label"],
            fontsize=8,
        )
    ax.set_title("Issuance Probability Cliff Curve")
    ax.set_xlabel("Weakest-Link Institutional Quality (%)")
    ax.set_ylabel("Issuance Probability (%)")
    ax.set_ylim(0, 100)
    _save_figure(fig, output_dir, "issuance_probability_cliff_curve")


def _chart_waterfall(summary_df: pd.DataFrame, output_dir: Path) -> None:
    base_mob = float(ARTEMIS_ANCHOR["base_mobilization_ratio"]["value"])
    premium = float(ARTEMIS_ANCHOR["blockchain_mobilization_premium"]["value"])
    penalty_weight = float(MODEL_PARAMETERS["mobilization_institutional_penalty_weight"]["value"])
    base_risk = float(ARTEMIS_ANCHOR["base_risk_premium_bps"]["value"])
    risk_scale = float(MODEL_PARAMETERS["risk_premium_bps_scale"]["value"])

    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    fig, axes = plt.subplots(1, len(chart_df), figsize=(20, 4), sharey=True)
    if len(chart_df) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, chart_df.iterrows()):
        institutional_quality = float(row["institutional_quality"])
        risk_value = float(row["median_risk_premium_bps"])
        increments = [
            base_mob,
            premium * institutional_quality,
            -(1.0 - institutional_quality) * penalty_weight,
            -((risk_value - base_risk) / risk_scale),
        ]
        labels = ["Base", "Blockchain premium", "Institutional penalty", "Risk penalty", "Final"]
        starts = [0.0, increments[0], increments[0] + increments[1], increments[0] + increments[1] + increments[2]]
        colors = ["#1f3b73", "#2e6f95", "#b22222", "#d2691e"]
        for idx, inc in enumerate(increments):
            ax.bar(idx, inc, bottom=starts[idx], color=colors[idx], width=0.7)
        final_value = increments[0] + increments[1] + increments[2] + increments[3]
        ax.bar(4, final_value, color="#4c956c", width=0.7)
        ax.set_xticks(np.arange(5), labels, rotation=45, ha="right")
        ax.set_title(row["scenario_label"], fontsize=9)
    fig.suptitle("Stress Waterfall: Mobilization Components", y=1.05)
    fig.tight_layout()
    _save_figure(fig, output_dir, "stress_waterfall_mobilization")


def _chart_severity_map(summary_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bubble_sizes = summary_df["median_time_to_issuance_months"] * 20.0
    ax.scatter(
        summary_df["france_integrity"] * 100.0,
        summary_df["artemis_integrity"] * 100.0,
        s=bubble_sizes,
        alpha=0.6,
        color="#2e6f95",
        edgecolor="black",
    )
    for _, row in summary_df.iterrows():
        ax.text(
            row["france_integrity"] * 100.0 + 0.6,
            row["artemis_integrity"] * 100.0 + 0.6,
            row["scenario_label"],
            fontsize=8,
        )
    ax.set_title("Scenario Severity Map")
    ax.set_xlabel("France Institutional Integrity (%)")
    ax.set_ylabel("ARTEMIS Operational Integrity (%)")
    _save_figure(fig, output_dir, "scenario_severity_map")
