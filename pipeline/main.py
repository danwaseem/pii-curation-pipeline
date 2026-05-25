"""
End-to-end PII curation pipeline.

Orchestrates all seven stages in sequence:
    1. Ingest       – multi-format data loading
    2. PII Detection – regex + spaCy NER
    3. Anonymization – hash-based pseudonymization and redaction
    4. Quality Scoring – vectorised four-dimension scoring
    5. Lineage Logging – per-record audit trail
    6. Quality Report – aggregate JSON summary
    7. Parquet Export – serialisation-safe processed dataset

Run:
    python -m pipeline.main
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.anonymizer import Anonymizer
from pipeline.ingest import (
    CSVAdapter,
    IngestOrchestrator,
    PlainTextAdapter,
    SlackAdapter,
)
from pipeline.lineage_logger import LineageLogger
from pipeline.pii_detector import PIIDetector
from pipeline.quality_scorer import QualityScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback: sets → sorted list, else str()."""
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def _to_json_str(obj: Any) -> str:
    """Serialize *obj* to a compact JSON string, handling sets and datetimes."""
    return json.dumps(obj, default=_json_default, ensure_ascii=False)


def _serialize_complex_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns that Parquet cannot serialise to JSON strings.

    Parquet / pyarrow struggles with Python sets, heterogeneous lists-of-dicts,
    and nested dicts with varying schemas.  Encoding as JSON strings sidesteps
    all of those issues while keeping the data fully recoverable.

    Columns converted (when present):
        raw_metadata, pii_detections, pii_types_found,
        anonymization_log, strategies_used
    """
    COMPLEX_COLS = [
        "raw_metadata",
        "pii_detections",
        "pii_types_found",
        "anonymization_log",
        "strategies_used",
    ]
    out = df.copy()
    for col in COMPLEX_COLS:
        if col in out.columns:
            out[col] = out[col].map(_to_json_str)
    return out


def _ensure_dirs(*paths: str) -> None:
    """Create every directory in *paths* (and their parents) if needed."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: dict) -> dict:
    """
    Run the full PII curation pipeline end-to-end.

    Parameters
    ----------
    config : dict
        Required keys
        -------------
        ``sources``
            dict mapping **filepath** (str) → **DataAdapter subclass** (type).
            Example::

                {
                    "data/raw/slack_messages.json": SlackAdapter,
                    "data/raw/project_records.csv": CSVAdapter,
                    "data/raw/code_comments.txt":   PlainTextAdapter,
                }

        Optional keys
        -------------
        ``quality_threshold``     float, default 0.8
        ``lineage_dir``           str,   default "data/lineage"
        ``quality_report_path``   str,   default "data/quality_reports/quality_report.json"
        ``parquet_path``          str,   default "data/processed/curated_dataset.parquet"

    Returns
    -------
    dict
        Run summary::

            {
                "total_records_ingested":           int,
                "pii_detected_in":                  int,
                "records_passed_quality_threshold": int,
                "quality_report_path":              str,
                "lineage_log_path":                 str,
                "parquet_path":                     str,
                "elapsed_seconds":                  float,
            }
    """
    t_start = time.perf_counter()

    # ── Resolve config values ────────────────────────────────────────────────
    quality_threshold: float = float(config.get("quality_threshold", 0.8))
    lineage_dir: str = config.get("lineage_dir", "data/lineage")
    quality_report_path: str = config.get(
        "quality_report_path",
        "data/quality_reports/quality_report.json",
    )
    parquet_path: str = config.get(
        "parquet_path",
        "data/processed/curated_dataset.parquet",
    )

    # Guarantee all output directories exist before any writes
    _ensure_dirs(
        lineage_dir,
        str(Path(quality_report_path).parent),
        str(Path(parquet_path).parent),
    )

    # ── Step 1: Ingest ───────────────────────────────────────────────────────
    print("[1/7] Ingesting data sources ...")
    orchestrator = IngestOrchestrator()
    df: pd.DataFrame = orchestrator.ingest_all(config["sources"])
    n_records = len(df)
    print(f"      {n_records} records loaded from {len(config['sources'])} source(s).")

    # ── Step 2: PII Detection ─────────────────────────────────────────────────
    print("[2/7] Detecting PII (regex + spaCy NER) ...")
    detector = PIIDetector()
    df = detector.detect_batch(df, "text")
    n_pii_records = int(df["has_pii"].sum())
    print(f"      PII found in {n_pii_records} of {n_records} records.")

    # ── Step 3: Anonymization ─────────────────────────────────────────────────
    print("[3/7] Anonymizing PII ...")
    anonymizer = Anonymizer()
    df = anonymizer.anonymize_batch(df)
    total_replacements = int(df["anonymization_count"].sum())
    print(f"      {total_replacements} PII spans replaced across all records.")

    # ── Step 4: Quality Scoring ───────────────────────────────────────────────
    print("[4/7] Scoring record quality ...")
    scorer = QualityScorer()
    df = scorer.score_batch(df)
    n_passed = int((df["quality_score"] >= quality_threshold).sum())
    mean_qs = float(df["quality_score"].mean())
    print(
        f"      {n_passed}/{n_records} records passed threshold "
        f"{quality_threshold} (mean score: {mean_qs:.3f})."
    )

    # ── Step 5: Lineage Logging ───────────────────────────────────────────────
    print("[5/7] Logging lineage events ...")
    logger = LineageLogger(output_dir=lineage_dir)

    # Capture a single timestamp for the detection and anonymization stages
    # (individual per-row timestamps would add noise without adding auditability
    # since all rows were processed in the same batch call above).
    detection_ts = _now_iso()
    anonymization_ts = _now_iso()

    for _, row in df.iterrows():
        record_id = str(row["record_id"])

        # — Ingestion event —
        logger.log_ingestion(
            record_id=record_id,
            source_format=str(row["source_format"]),
            source_file=str(row["source_file"]),
            timestamp=row["timestamp"],
            text_length=len(str(row.get("text") or "")),
        )

        # — PII detection event —
        pii_types = row.get("pii_types_found") or set()
        pii_count = int(row.get("pii_count") or 0)
        logger.log_pii_detection(
            record_id=record_id,
            pii_types_found=pii_types,
            pii_count=pii_count,
            detection_timestamp=detection_ts,
        )

        # — Anonymization events — one entry per replaced PII span —
        anon_log: list[dict] = row.get("anonymization_log") or []
        for entry in anon_log:
            logger.log_anonymization(
                record_id=record_id,
                strategy=entry["strategy"],
                pii_type=entry["pii_type"],
                original_length=len(entry.get("original_text") or ""),
                replacement=entry["replacement"],
                anonymization_timestamp=anonymization_ts,
            )

        # — Quality scoring event —
        logger.log_quality_score(
            record_id=record_id,
            quality_score=float(row["quality_score"]),
            completeness_score=float(row["completeness_score"]),
            pii_removal_confidence_score=float(row["pii_removal_confidence_score"]),
            passed_threshold=bool(float(row["quality_score"]) >= quality_threshold),
        )

    # Persist all events; buffer is cleared automatically after flush
    n_events = len(logger)          # capture count before flush zeroes it
    lineage_log_path = logger.flush()
    print(f"      {n_events} events flushed to: {lineage_log_path}")

    # ── Step 6: Quality Report ────────────────────────────────────────────────
    print("[6/7] Generating quality report ...")
    scorer.generate_report(df, output_path=quality_report_path)
    print(f"      Report saved to: {quality_report_path}")

    # ── Step 7: Save Processed Parquet ────────────────────────────────────────
    print("[7/7] Saving processed dataset to Parquet ...")
    df_parquet = _serialize_complex_cols(df)
    df_parquet.to_parquet(parquet_path, index=False)
    print(f"      Dataset saved to: {parquet_path}")

    # ── Run summary ───────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start

    summary: dict[str, Any] = {
        "total_records_ingested": n_records,
        "pii_detected_in": n_pii_records,
        "records_passed_quality_threshold": n_passed,
        "quality_report_path": quality_report_path,
        "lineage_log_path": lineage_log_path,
        "parquet_path": parquet_path,
        "elapsed_seconds": round(elapsed, 2),
    }

    _print_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(summary: dict) -> None:
    """Print the human-readable run summary."""
    sep = "-" * 62
    print()
    print(sep)
    print(f"  Total records ingested:            {summary['total_records_ingested']}")
    print(f"  PII detected in:                   {summary['pii_detected_in']} records")
    print(f"  Records passed quality threshold:  {summary['records_passed_quality_threshold']}")
    print(f"  Quality report saved to:           {summary['quality_report_path']}")
    print(f"  Lineage log saved to:              {summary['lineage_log_path']}")
    print(f"  Dataset saved to:                  {summary['parquet_path']}")
    print(f"  Elapsed:                           {summary['elapsed_seconds']}s")
    print(sep)


# ---------------------------------------------------------------------------
# Default configuration and __main__ entry point
# ---------------------------------------------------------------------------

# Project root is two levels up from this file (pipeline/main.py → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG: dict = {
    "sources": {
        str(_PROJECT_ROOT / "data" / "raw" / "slack_messages.json"): SlackAdapter,
        str(_PROJECT_ROOT / "data" / "raw" / "project_records.csv"): CSVAdapter,
        str(_PROJECT_ROOT / "data" / "raw" / "code_comments.txt"): PlainTextAdapter,
    },
    "quality_threshold": 0.8,
    "lineage_dir": str(_PROJECT_ROOT / "data" / "lineage"),
    "quality_report_path": str(
        _PROJECT_ROOT / "data" / "quality_reports" / "quality_report.json"
    ),
    "parquet_path": str(
        _PROJECT_ROOT / "data" / "processed" / "curated_dataset.parquet"
    ),
}


if __name__ == "__main__":
    run_pipeline(DEFAULT_CONFIG)
