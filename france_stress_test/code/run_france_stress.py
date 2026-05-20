from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .artemis_anchor import anchor_table
    from .config import ALLOWED_COUNTRIES, AUDIT_DIR, DATA_DIR, FIGURES_DIR, REPORT_DIR, TABLES_DIR
    from .france_pathways import build_pathway_dataframe, build_pathway_integrity_heatmap, compute_france_readiness
    from .stress_charts import generate_france_figures
    from .stress_engine import run_france_stress_test
except ImportError:
    from artemis_anchor import anchor_table
    from config import ALLOWED_COUNTRIES, AUDIT_DIR, DATA_DIR, FIGURES_DIR, REPORT_DIR, TABLES_DIR
    from france_pathways import build_pathway_dataframe, build_pathway_integrity_heatmap, compute_france_readiness
    from stress_charts import generate_france_figures
    from stress_engine import run_france_stress_test


def _ensure_directories() -> None:
    for directory in (DATA_DIR, FIGURES_DIR, TABLES_DIR, AUDIT_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _validate_country_scope(pathway_df: pd.DataFrame) -> None:
    if "France" not in ALLOWED_COUNTRIES:
        raise ValueError("France must exist in the allowed-country list.")
    if pathway_df.empty:
        raise ValueError("France pathway table cannot be empty.")


def _validate_source_classification(pathway_df: pd.DataFrame, summary_df: pd.DataFrame, iterations_df: pd.DataFrame) -> None:
    for frame_name, frame in (
        ("pathway_df", pathway_df),
        ("summary_df", summary_df),
        ("iterations_df", iterations_df),
    ):
        if "source_type" not in frame.columns:
            raise ValueError(f"{frame_name} must include source_type.")
        if frame["source_type"].isna().any():
            raise ValueError(f"{frame_name} has missing source_type values.")
    if (pathway_df["source_type"] == "OBSERVED_DATA").any():
        raise ValueError("France pathway scores cannot be labeled OBSERVED_DATA in this version.")


def _write_audit_report(pathway_df: pd.DataFrame, summary_df: pd.DataFrame, audit_path: Path) -> None:
    pathway_note = f"France pathway weighted score: {pathway_df['weighted_score'].sum():.2f}."
    status_counts = summary_df["status"].value_counts().to_dict()
    lines = [
        "# France Stress Test Data Audit",
        "",
        "## Source Classification",
        "- France pathway scores are DOCUMENTED_AUTHOR_SCORE.",
        "- ARTEMIS anchor values mix DOCUMENTED_AUTHOR_SCORE and MODEL_ASSUMPTION.",
        "- Stress scenario shocks are MODEL_ASSUMPTION.",
        "- No stress scenario should be interpreted as an observed historical event.",
        "- The weakest-link principle is used because one failed pathway can block issuance.",
        "",
        "## Additional Validation Notes",
        f"- {pathway_note}",
        f"- Scenario status distribution: {status_counts}.",
        "- All simulation probabilities are clipped to [0, 1].",
    ]
    audit_path.write_text("\n".join(lines), encoding="utf-8")


def _write_model_summary(summary_df: pd.DataFrame, readiness_score: float, report_path: Path) -> None:
    worst = summary_df.sort_values("median_issuance_probability", ascending=True).iloc[0]
    lines = [
        "# France Stress Model Summary",
        "",
        f"France readiness score: {readiness_score:.2f}/100.",
        f"Worst issuance scenario: {worst['scenario_label']} ({worst['median_issuance_probability']:.2%} issuance probability).",
        "",
        '"Blockchain-enabled governance may improve credibility and reduce investor uncertainty, but its feasibility depends on institutional readiness. The weakest-link pathway is decisive: if data admissibility, legal enforceability, or regulatory classification fails, issuance probability can collapse even when the mobilization ratio remains near the target."',
        "",
        "This model is a structured simulation and feasibility tool. It does not prove real-world performance.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _validate_target_status_alignment(summary_df: pd.DataFrame) -> None:
    mismatches = summary_df.loc[summary_df["status"] != summary_df["target_status"], ["scenario_label", "status", "target_status"]]
    if not mismatches.empty:
        mismatch_text = ", ".join(
            f"{row['scenario_label']}: actual={row['status']} expected={row['target_status']}"
            for _, row in mismatches.iterrows()
        )
        raise ValueError(f"Scenario target-status mismatch detected. {mismatch_text}")


def run() -> None:
    _ensure_directories()
    pathway_df = build_pathway_dataframe()
    readiness = compute_france_readiness(pathway_df)
    _validate_country_scope(pathway_df)

    stress_result = run_france_stress_test()
    pathway_integrity_df = build_pathway_integrity_heatmap()
    anchor_df = anchor_table()
    _validate_source_classification(pathway_df, stress_result.summary, stress_result.iterations)
    _validate_target_status_alignment(stress_result.summary)

    pathway_df.to_csv(DATA_DIR / "france_pathway_scores.csv", index=False)
    stress_result.summary.to_csv(DATA_DIR / "france_stress_results.csv", index=False)
    stress_result.iterations.to_csv(DATA_DIR / "france_stress_iterations.csv", index=False)

    with pd.ExcelWriter(TABLES_DIR / "france_pathway_scores.xlsx") as writer:
        pathway_df.to_excel(writer, sheet_name="pathways", index=False)
        anchor_df.to_excel(writer, sheet_name="artemis_anchor", index=False)
    with pd.ExcelWriter(TABLES_DIR / "france_stress_summary.xlsx") as writer:
        stress_result.summary.to_excel(writer, sheet_name="summary", index=False)
        pathway_integrity_df.to_excel(writer, sheet_name="pathway_integrity", index=False)

    _write_audit_report(pathway_df, stress_result.summary, AUDIT_DIR / "france_data_audit_report.md")
    _write_model_summary(
        stress_result.summary,
        readiness_score=readiness.score,
        report_path=REPORT_DIR / "france_stress_model_summary.md",
    )
    generate_france_figures(
        pathway_df=pathway_df,
        summary_df=stress_result.summary,
        iterations_df=stress_result.iterations,
        pathway_integrity_df=pathway_integrity_df,
        output_dir=FIGURES_DIR,
    )

    print("France stress test outputs generated.")
    print(f"Readiness score: {readiness.score:.2f} ({readiness.classification})")
    print(f"Output directory: {DATA_DIR.parents[0]}")


if __name__ == "__main__":
    run()
