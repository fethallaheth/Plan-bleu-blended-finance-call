from __future__ import annotations

import pandas as pd

try:
    from .config import AUDIT_DIR, DATA_DIR, FIGURES_DIR, REPORT_DIR, TABLES_DIR
    from .readiness_engine import build_country_readiness_scores, optional_dataset_validation_placeholder, source_coverage_table
    from .transfer_charts import generate_transferability_figures
    from .transfer_engine import run_transferability_model
except ImportError:
    from config import AUDIT_DIR, DATA_DIR, FIGURES_DIR, REPORT_DIR, TABLES_DIR
    from readiness_engine import build_country_readiness_scores, optional_dataset_validation_placeholder, source_coverage_table
    from transfer_charts import generate_transferability_figures
    from transfer_engine import run_transferability_model


def _ensure_directories() -> None:
    for directory in (DATA_DIR, FIGURES_DIR, TABLES_DIR, AUDIT_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _write_audit_report(audit_text: str) -> None:
    lines = [
        "# Transferability Data Audit",
        "",
        "## Source Classification",
        "- Country readiness scores are DOCUMENTED_AUTHOR_SCORE in this version.",
        "- They can later be validated/replaced using WGI, EGDI, IDI, and B-READY.",
        "- The scores are feasibility proxies, not official rankings.",
        "- France is used as the benchmark.",
        "- Countries with weaker scores require phased implementation and capacity-building.",
        "",
        "## Dataset Validation Hook",
        f"- {audit_text}",
    ]
    (AUDIT_DIR / "transferability_data_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_model_summary() -> None:
    lines = [
        "# Transferability Model Summary",
        "",
        '"France is the benchmark. Transferability to other Mediterranean countries depends on regulatory clarity, supervisory capacity, legal enforceability, public-finance compatibility, digital infrastructure, environmental monitoring, and sustainable-finance market maturity. Countries with weaker readiness scores require phased implementation and capacity-building before a full blockchain-enabled blended Blue Bond architecture can be deployed."',
        "",
        "This model is a structured simulation and feasibility tool. It does not prove real-world performance.",
    ]
    (REPORT_DIR / "transferability_model_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _validate_source_classification(readiness_df: pd.DataFrame, transfer_df: pd.DataFrame, source_df: pd.DataFrame) -> None:
    for frame_name, frame in (
        ("readiness_df", readiness_df),
        ("transfer_df", transfer_df),
    ):
        if "source_type" not in frame.columns:
            raise ValueError(f"{frame_name} must include source_type.")
        if frame["source_type"].isna().any():
            raise ValueError(f"{frame_name} has missing source_type values.")
    required_source_columns = {"OBSERVED_DATA", "DERIVED_DATA", "DOCUMENTED_AUTHOR_SCORE", "MODEL_ASSUMPTION"}
    if not required_source_columns.issubset(set(source_df.columns)):
        raise ValueError("source coverage table is missing required source-type columns.")


def run() -> None:
    _ensure_directories()
    readiness_df = build_country_readiness_scores()
    transfer_result = run_transferability_model(readiness_df)
    source_df = source_coverage_table(readiness_df)
    audit_hook = optional_dataset_validation_placeholder()
    _validate_source_classification(transfer_result.readiness_scores, transfer_result.transferability, source_df)

    transfer_result.readiness_scores.to_csv(DATA_DIR / "country_readiness_scores.csv", index=False)
    transfer_result.transferability.to_csv(DATA_DIR / "transferability_results.csv", index=False)

    with pd.ExcelWriter(TABLES_DIR / "country_readiness_scores.xlsx") as writer:
        transfer_result.readiness_scores.to_excel(writer, sheet_name="scores", index=False)
    with pd.ExcelWriter(TABLES_DIR / "transferability_summary.xlsx") as writer:
        transfer_result.transferability.to_excel(writer, sheet_name="summary", index=False)
        source_df.to_excel(writer, sheet_name="source_coverage", index=False)

    _write_audit_report(audit_hook)
    _write_model_summary()
    generate_transferability_figures(
        readiness_df=transfer_result.readiness_scores,
        transferability_df=transfer_result.transferability,
        source_coverage_df=source_df,
        output_dir=FIGURES_DIR,
    )

    print("Transferability outputs generated.")
    print(f"Output directory: {DATA_DIR.parents[0]}")


if __name__ == "__main__":
    run()
