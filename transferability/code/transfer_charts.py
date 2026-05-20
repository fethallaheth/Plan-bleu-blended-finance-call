from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .config import READINESS_PILLARS, TRANSFER_PARAMETERS
except ImportError:
    from config import READINESS_PILLARS, TRANSFER_PARAMETERS


FORMATS = ("png", "svg")
DPI = 300


def generate_transferability_figures(
    readiness_df: pd.DataFrame,
    transferability_df: pd.DataFrame,
    source_coverage_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    _chart_country_readiness_ranking(readiness_df, output_dir)
    _chart_transferability_matrix(readiness_df, output_dir)
    _chart_readiness_vs_mobilization(transferability_df, output_dir)
    _chart_readiness_vs_issuance(transferability_df, output_dir)
    _chart_transferability_roadmap(transferability_df, output_dir)
    _chart_binding_constraint_heatmap(readiness_df, output_dir)
    _chart_france_benchmark_comparison(transferability_df, output_dir)
    _chart_source_coverage(source_coverage_df, output_dir)
    _chart_transferability_grade_table(transferability_df, output_dir)
    _chart_readiness_mobilization_risk_bubble(transferability_df, output_dir)
    _chart_country_pillar_radar(readiness_df, output_dir)


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    for extension in FORMATS:
        fig.savefig(output_dir / f"{name}.{extension}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _wrap_label(value: str, width: int = 16) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def _annotate_points(ax: plt.Axes, xs: pd.Series, ys: pd.Series, labels: pd.Series, dx: float = 0.6, dy: float = 0.02) -> None:
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


def _chart_country_readiness_ranking(readiness_df: pd.DataFrame, output_dir: Path) -> None:
    chart_df = readiness_df.sort_values("readiness_score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(chart_df["country"], chart_df["readiness_score"], color="#2e6f95")
    for bar in bars:
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.1f}", va="center")
    ax.set_xlim(0, 100)
    ax.set_title("Country Readiness Ranking")
    ax.set_xlabel("Readiness Score")
    _save_figure(fig, output_dir, "country_readiness_ranking")


def _chart_transferability_matrix(readiness_df: pd.DataFrame, output_dir: Path) -> None:
    matrix = readiness_df[["country", *READINESS_PILLARS, "readiness_score"]].set_index("country")
    matrix = matrix.rename(columns={column: _wrap_label(column.replace("_", " ").title(), width=14) for column in matrix.columns})
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5, cbar_kws={"label": "Score"}, ax=ax)
    ax.set_title("Transferability Matrix Heatmap")
    ax.set_xlabel("Readiness Pillars and Final Score")
    ax.set_ylabel("Country")
    _save_figure(fig, output_dir, "transferability_matrix_heatmap")


def _chart_readiness_vs_mobilization(transfer_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(transfer_df["readiness_score"], transfer_df["expected_mobilization_ratio"], color="#1f3b73", s=80)
    _annotate_points(ax, transfer_df["readiness_score"], transfer_df["expected_mobilization_ratio"], transfer_df["country"])
    ax.axhline(4.0, color="crimson", linestyle="--", linewidth=1.1)
    for threshold in (70, 55, 40):
        ax.axvline(threshold, color="gray", linestyle=":", linewidth=1.0)
    ax.set_title("Readiness vs Expected Mobilization")
    ax.set_xlabel("Readiness Score")
    ax.set_ylabel("Expected Mobilization Ratio")
    _save_figure(fig, output_dir, "readiness_vs_expected_mobilization")


def _chart_readiness_vs_issuance(transfer_df: pd.DataFrame, output_dir: Path) -> None:
    x = np.linspace(0, 100, 400)
    slope = float(TRANSFER_PARAMETERS["issuance_sigmoid_slope"]["value"])
    y = 1.0 / (1.0 + np.exp(-((x - 50.0) * slope)))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y * 100.0, color="#2e6f95", linewidth=2)
    ax.scatter(transfer_df["constraint_adjusted_score"], transfer_df["issuance_probability"] * 100.0, color="#b22222", s=60)
    _annotate_points(
        ax,
        transfer_df["constraint_adjusted_score"],
        transfer_df["issuance_probability"] * 100.0,
        transfer_df["country"],
        dx=0.9,
        dy=0.03,
    )
    ax.set_title("Constraint-Adjusted Score vs Issuance Probability")
    ax.set_xlabel("Constraint-Adjusted Transferability Score")
    ax.set_ylabel("Issuance Probability (%)")
    ax.set_ylim(0, 100)
    _save_figure(fig, output_dir, "readiness_vs_issuance_probability")


def _chart_transferability_roadmap(transfer_df: pd.DataFrame, output_dir: Path) -> None:
    order = transfer_df.sort_values("adaptation_time_months", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    tracks = {
        "near-term": "#2e6f95",
        "medium-term": "#6ca6c1",
        "long-term": "#f4a259",
        "indirect/very long-term": "#b22222",
    }
    for idx, (_, row) in enumerate(order.iterrows()):
        ax.barh(
            y=idx,
            width=row["adaptation_time_months"],
            color=tracks.get(row["adaptation_track"], "#2e6f95"),
            edgecolor="black",
        )
    ax.set_yticks(np.arange(len(order)), order["country"].to_list())
    ax.set_xlabel("Adaptation Timeframe (months)")
    ax.set_title("Transferability Roadmap by Adaptation Time")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    _save_figure(fig, output_dir, "transferability_roadmap")


def _chart_binding_constraint_heatmap(readiness_df: pd.DataFrame, output_dir: Path) -> None:
    matrix = readiness_df[["country", *READINESS_PILLARS]].set_index("country")
    matrix = matrix.rename(columns={column: _wrap_label(column.replace("_", " ").title(), width=14) for column in matrix.columns})
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, cbar_kws={"label": "Pillar Score"}, ax=ax)
    for row_idx, (_, row) in enumerate(readiness_df.iterrows()):
        constraint_col = READINESS_PILLARS.index(row["main_binding_constraint"])
        ax.add_patch(plt.Rectangle((constraint_col, row_idx), 1, 1, fill=False, edgecolor="crimson", linewidth=2))
    ax.set_title("Binding Constraint Heatmap")
    _save_figure(fig, output_dir, "binding_constraint_heatmap")


def _chart_france_benchmark_comparison(transfer_df: pd.DataFrame, output_dir: Path) -> None:
    chart_df = transfer_df.copy()
    positions = np.arange(len(chart_df))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(positions, chart_df["readiness_gap_vs_france"], color="#1f3b73")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Readiness Gap Relative to France Benchmark")
    ax.set_ylabel("Readiness Gap (points)")
    ax.set_xticks(positions)
    ax.set_xticklabels(chart_df["country"], rotation=25, ha="right")
    _save_figure(fig, output_dir, "france_benchmark_comparison")


def _chart_source_coverage(source_df: pd.DataFrame, output_dir: Path) -> None:
    categories = ["OBSERVED_DATA", "DERIVED_DATA", "DOCUMENTED_AUTHOR_SCORE", "MODEL_ASSUMPTION"]
    colors = ["#bdbdbd", "#8da0cb", "#2e6f95", "#fc8d62"]
    fig, ax = plt.subplots(figsize=(11.5, 6))
    bottoms = np.zeros(len(source_df))
    for category, color in zip(categories, colors):
        values = source_df[category].to_numpy()
        ax.bar(source_df["country"], values, bottom=bottoms, label=category, color=color)
        bottoms += values
    ax.set_ylim(0, 100)
    ax.set_title("Readiness Data Source Coverage")
    ax.set_ylabel("Share of Input Structure (%)")
    ax.legend(frameon=False, fontsize=8, ncols=2, loc="upper right")
    ax.tick_params(axis="x", rotation=25)
    _save_figure(fig, output_dir, "readiness_data_source_coverage")


def _chart_transferability_grade_table(transfer_df: pd.DataFrame, output_dir: Path) -> None:
    display = transfer_df[
        [
            "country",
            "constraint_adjusted_score",
            "transferability_grade",
            "expected_mobilization_ratio",
            "expected_risk_premium_bps",
            "issuance_probability",
            "adaptation_time_months",
            "main_binding_constraint",
        ]
    ].copy()
    display["issuance_probability"] = (display["issuance_probability"] * 100.0).round(1)
    display["main_binding_constraint"] = display["main_binding_constraint"].str.replace("_", " ", regex=False)
    display = display.rename(
        columns={
            "constraint_adjusted_score": "transfer_score",
            "transferability_grade": "grade",
            "expected_mobilization_ratio": "mobilization",
            "expected_risk_premium_bps": "risk_bps",
            "issuance_probability": "issuance_%",
            "adaptation_time_months": "adapt_months",
            "main_binding_constraint": "binding_constraint",
        }
    )
    display = display.round(
        {
            "transfer_score": 2,
            "mobilization": 3,
            "risk_bps": 1,
            "adapt_months": 1,
        }
    )
    display = display.rename(
        columns={
            "country": "Country",
            "transfer_score": "Transfer score",
            "grade": "Grade",
            "mobilization": "Mobilization",
            "risk_bps": "Risk (bps)",
            "issuance_%": "Issuance (%)",
            "adapt_months": "Adapt. months",
            "binding_constraint": "Binding constraint",
        }
    )
    fig, ax = plt.subplots(figsize=(17.5, 4.8))
    ax.axis("off")
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.55)
    ax.set_title("Transferability Grade Table", pad=10)
    _save_figure(fig, output_dir, "transferability_grade_table")


def _chart_readiness_mobilization_risk_bubble(transfer_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bubble_sizes = transfer_df["expected_risk_premium_bps"] * 0.5
    ax.scatter(
        transfer_df["readiness_score"],
        transfer_df["expected_mobilization_ratio"],
        s=bubble_sizes,
        alpha=0.55,
        color="#2e6f95",
        edgecolor="black",
    )
    _annotate_points(ax, transfer_df["readiness_score"], transfer_df["expected_mobilization_ratio"], transfer_df["country"])
    ax.set_title("Readiness, Mobilization, and Risk Premium")
    ax.set_xlabel("Readiness Score")
    ax.set_ylabel("Expected Mobilization Ratio")
    _save_figure(fig, output_dir, "readiness_mobilization_risk_bubble")


def _chart_country_pillar_radar(readiness_df: pd.DataFrame, output_dir: Path) -> None:
    selected = ["France", "Spain", "Turkey", "Algeria", "Lebanon"]
    chart_df = readiness_df.set_index("country").loc[selected, READINESS_PILLARS]
    angles = np.linspace(0, 2 * np.pi, len(READINESS_PILLARS), endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])
    fig, ax = plt.subplots(figsize=(9.5, 8.5), subplot_kw={"projection": "polar"})
    for country in selected:
        values = chart_df.loc[country].to_numpy()
        values_closed = np.concatenate([values, values[:1]])
        ax.plot(angles_closed, values_closed, linewidth=1.7, label=country)
    ax.set_xticks(angles)
    ax.set_xticklabels([_wrap_label(pillar.replace("_", " ").title(), width=14) for pillar in READINESS_PILLARS], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title("Country Pillar Radar Comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.12), frameon=False)
    _save_figure(fig, output_dir, "country_pillar_radar_comparison")
