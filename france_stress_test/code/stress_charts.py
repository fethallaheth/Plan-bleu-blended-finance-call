from __future__ import annotations

from pathlib import Path
import textwrap

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


def _wrap_label(value: str, width: int = 16) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def _annotate_points(ax: plt.Axes, xs: pd.Series, ys: pd.Series, labels: pd.Series, dx: float = 0.8, dy: float = 0.03) -> None:
    offsets = [(-1, 1), (1, 1), (-1, -1), (1, -1), (0, 1)]
    y_span = float(max(ys.max() - ys.min(), 1.0))
    for idx, (x_value, y_value, label) in enumerate(zip(xs, ys, labels)):
        x_sign, y_sign = offsets[idx % len(offsets)]
        ax.annotate(
            str(label),
            xy=(x_value, y_value),
            xytext=(x_value + (x_sign * dx), y_value + (y_sign * dy * y_span)),
            textcoords="data",
            fontsize=8,
            ha="left" if x_sign >= 0 else "right",
            va="bottom" if y_sign >= 0 else "top",
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#7a7a7a"},
        )


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
    display_df = display_df.rename(
        columns={
            "scenario_label": "scenario",
            "institutional_quality": "weakest_link_%",
            "median_mobilization_ratio": "mobilization",
            "median_risk_premium_bps": "risk_bps",
            "median_governance_cost_pct": "gov_cost_%",
            "median_issuance_probability": "issuance_%",
            "median_pes_achievement_eur": "pes_eur",
            "median_data_admissibility_probability": "admissibility_%",
            "median_time_to_issuance_months": "time_months",
        }
    )
    display_df["scenario"] = display_df["scenario"].map(lambda value: _wrap_label(value, width=20))
    display_df["weakest_link_%"] = (display_df["weakest_link_%"] * 100.0).round(1)
    display_df["gov_cost_%"] = (display_df["gov_cost_%"] * 100.0).round(2)
    display_df["issuance_%"] = (display_df["issuance_%"] * 100.0).round(1)
    display_df["admissibility_%"] = (display_df["admissibility_%"] * 100.0).round(1)
    display_df = display_df.round(
        {
            "mobilization": 3,
            "risk_bps": 1,
            "pes_eur": 0,
            "time_months": 1,
        }
    )
    display_df = display_df.rename(
        columns={
            "scenario": "Scenario",
            "weakest_link_%": "Weakest link (%)",
            "mobilization": "Mobilization",
            "risk_bps": "Risk (bps)",
            "gov_cost_%": "Gov cost (%)",
            "issuance_%": "Issuance (%)",
            "pes_eur": "PES (EUR)",
            "admissibility_%": "Admissibility (%)",
            "time_months": "Time (months)",
            "status": "Status",
        }
    )

    fig, ax = plt.subplots(figsize=(19.5, 4.5))
    ax.axis("off")
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.75)
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
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([_wrap_label(label, width=18) for label in order], rotation=0, ha="center")
    _save_figure(fig, output_dir, "stress_mobilization_boxplot")


def _chart_issuance_collapse(summary_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    positions = np.arange(len(chart_df))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(positions, chart_df["median_issuance_probability"] * 100.0, color="#1f3b73")
    ax.set_ylim(0, 100)
    ax.set_title("Bond Issuance Probability Collapse by Scenario")
    ax.set_ylabel("Issuance Probability (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels([_wrap_label(label, width=18) for label in chart_df["scenario_label"]], rotation=0, ha="center")
    _save_figure(fig, output_dir, "stress_issuance_probability_collapse")


def _chart_risk_premium(summary_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    values = chart_df["median_risk_premium_bps"].to_numpy()
    lower = values - chart_df["risk_premium_p10"].to_numpy()
    upper = chart_df["risk_premium_p90"].to_numpy() - values
    positions = np.arange(len(chart_df))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(positions, values, color="#2e6f95")
    ax.errorbar(positions, values, yerr=[lower, upper], fmt="none", capsize=5, color="black")
    ax.set_title("Risk Premium by Stress Scenario")
    ax.set_ylabel("Risk Premium (bps)")
    ax.set_xticks(positions)
    ax.set_xticklabels([_wrap_label(label, width=18) for label in chart_df["scenario_label"]], rotation=0, ha="center")
    _save_figure(fig, output_dir, "stress_risk_premium_by_scenario")


def _chart_governance_cost(summary_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    values = chart_df["median_governance_cost_pct"].to_numpy() * 100.0
    lower = values - chart_df["governance_cost_p10"].to_numpy() * 100.0
    upper = chart_df["governance_cost_p90"].to_numpy() * 100.0 - values
    positions = np.arange(len(chart_df))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(positions, values, color="#4c956c")
    ax.errorbar(positions, values, yerr=[lower, upper], fmt="none", capsize=5, color="black")
    ax.set_title("Governance Cost by Stress Scenario")
    ax.set_ylabel("Governance Cost (% of project value)")
    ax.set_xticks(positions)
    ax.set_xticklabels([_wrap_label(label, width=18) for label in chart_df["scenario_label"]], rotation=0, ha="center")
    _save_figure(fig, output_dir, "stress_governance_cost_by_scenario")


def _chart_pes(summary_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    positions = np.arange(len(chart_df))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(positions, chart_df["median_pes_achievement_eur"], color="#6ca6c1")
    ax.axhline(2_000_000, color="crimson", linestyle="--", linewidth=1.2)
    ax.set_title("PES Achievement Under Stress")
    ax.set_ylabel("PES Achievement (EUR)")
    ax.set_xticks(positions)
    ax.set_xticklabels([_wrap_label(label, width=18) for label in chart_df["scenario_label"]], rotation=0, ha="center")
    _save_figure(fig, output_dir, "stress_pes_achievement")


def _chart_data_admissibility(summary_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    colors = ["#2e6f95"] * len(chart_df)
    highlight = "B: Greenwashing / Oracle Data Challenge"
    if highlight in chart_df["scenario_label"].values:
        colors[chart_df["scenario_label"].to_list().index(highlight)] = "#b22222"
    positions = np.arange(len(chart_df))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(positions, chart_df["median_data_admissibility_probability"] * 100.0, color=colors)
    ax.set_title("Data Admissibility Probability Under Stress")
    ax.set_ylabel("Data Admissibility (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(positions)
    ax.set_xticklabels([_wrap_label(label, width=18) for label in chart_df["scenario_label"]], rotation=0, ha="center")
    _save_figure(fig, output_dir, "stress_data_admissibility")


def _chart_time_to_issuance(summary_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    positions = np.arange(len(chart_df))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(positions, chart_df["median_time_to_issuance_months"], color="#1f3b73")
    ax.axhline(30.0, color="crimson", linestyle="--", linewidth=1.2)
    ax.set_title("Time to Issuance Under Stress")
    ax.set_ylabel("Months")
    ax.set_xticks(positions)
    ax.set_xticklabels([_wrap_label(label, width=18) for label in chart_df["scenario_label"]], rotation=0, ha="center")
    ax.legend(["30-month reference line"], frameon=False, loc="upper left")
    _save_figure(fig, output_dir, "stress_time_to_issuance")


def _chart_pathway_integrity_heatmap(pathway_integrity_df: pd.DataFrame, output_dir: Path) -> None:
    chart_df = pathway_integrity_df.set_index("pathway")
    rename_map = {key: payload["label"] for key, payload in FRANCE_STRESS_SCENARIOS.items()}
    chart_df = chart_df.rename(columns=rename_map)
    chart_df.index = [_wrap_label(label, width=18) for label in chart_df.index]
    chart_df.columns = [_wrap_label(label, width=18) for label in chart_df.columns]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sns.heatmap(chart_df, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5, cbar_kws={"label": "Integrity Score"}, ax=ax)
    ax.set_title("Pathway Integrity Heatmap by Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("France Pathways")
    _save_figure(fig, output_dir, "pathway_integrity_heatmap")


def _chart_dashboard(summary_df: pd.DataFrame, iterations_df: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_labels()
    chart_df = summary_df.set_index("scenario_label").loc[order].reset_index()
    positions = np.arange(len(chart_df))
    wrapped_labels = [_wrap_label(label, width=18) for label in order]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.boxplot(data=iterations_df, x="scenario_label", y="mobilization_ratio", order=order, showfliers=False, ax=axes[0, 0], color="#6ca6c1")
    axes[0, 0].axhline(4.0, color="crimson", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("Mobilization Ratio")
    axes[0, 0].set_xticks(positions)
    axes[0, 0].set_xticklabels(wrapped_labels, rotation=0, ha="center")

    axes[0, 1].bar(positions, chart_df["median_issuance_probability"] * 100.0, color="#1f3b73")
    axes[0, 1].set_title("Issuance Probability (%)")
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].set_xticks(positions)
    axes[0, 1].set_xticklabels(wrapped_labels, rotation=0, ha="center")

    axes[1, 0].bar(positions, chart_df["median_risk_premium_bps"], color="#2e6f95")
    axes[1, 0].set_title("Risk Premium (bps)")
    axes[1, 0].set_xticks(positions)
    axes[1, 0].set_xticklabels(wrapped_labels, rotation=0, ha="center")

    axes[1, 1].bar(positions, chart_df["median_time_to_issuance_months"], color="#4c956c")
    axes[1, 1].axhline(30.0, color="crimson", linestyle="--", linewidth=1.0)
    axes[1, 1].set_title("Time to Issuance (months)")
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels(wrapped_labels, rotation=0, ha="center")

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
    xs = summary_df["institutional_quality"] * 100.0
    ys = summary_df["median_issuance_probability"] * 100.0
    ax.scatter(xs, ys, color="#b22222")
    _annotate_points(ax, xs, ys, summary_df["scenario_label"], dx=2.0, dy=0.05)
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
    fig, axes = plt.subplots(1, len(chart_df), figsize=(22, 5.5), sharey=True)
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
        labels = ["Base", "BC +", "Inst -", "Risk -", "Final"]
        starts = [0.0, increments[0], increments[0] + increments[1], increments[0] + increments[1] + increments[2]]
        colors = ["#1f3b73", "#2e6f95", "#b22222", "#d2691e"]
        for idx, inc in enumerate(increments):
            ax.bar(idx, inc, bottom=starts[idx], color=colors[idx], width=0.7)
        final_value = increments[0] + increments[1] + increments[2] + increments[3]
        ax.bar(4, final_value, color="#4c956c", width=0.7)
        ax.set_xticks(np.arange(5), labels)
        ax.set_title(_wrap_label(row["scenario_label"], width=18), fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
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
    xs = summary_df["france_integrity"] * 100.0
    ys = summary_df["artemis_integrity"] * 100.0
    _annotate_points(ax, xs, ys, summary_df["scenario_label"], dx=2.0, dy=0.05)
    ax.set_title("Scenario Severity Map")
    ax.set_xlabel("France Institutional Integrity (%)")
    ax.set_ylabel("ARTEMIS Operational Integrity (%)")
    _save_figure(fig, output_dir, "scenario_severity_map")
