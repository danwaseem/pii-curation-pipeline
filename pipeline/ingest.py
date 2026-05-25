"""
Multi-format ingestion system for the PII curation pipeline.

Run directly to ingest all three synthetic raw data files:
    python -m pipeline.ingest
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified schema
# ---------------------------------------------------------------------------

UNIFIED_COLUMNS: list[str] = [
    "record_id",
    "source_format",
    "source_file",
    "timestamp",
    "text",
    "raw_metadata",
]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class DataAdapter(ABC):
    """Abstract base class for format-specific ingestion adapters."""

    source_format: ClassVar[str] = "unknown"

    @abstractmethod
    def load(self, filepath: str) -> pd.DataFrame:
        """Load *filepath* and return a DataFrame conforming to the unified schema."""

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Return True if *df* satisfies this adapter's schema invariants."""


# ---------------------------------------------------------------------------
# SlackAdapter
# ---------------------------------------------------------------------------

class SlackAdapter(DataAdapter):
    """Ingests a Slack-style JSON export (array of message objects).

    Unified schema mapping
    ----------------------
    record_id     : generated UUID
    source_format : "slack"
    source_file   : resolved filepath
    timestamp     : parsed ISO-8601 datetime
    text          : message body
    raw_metadata  : {"channel": ..., "username": ..., "user_id": ...}
    """

    source_format: ClassVar[str] = "slack"

    def load(self, filepath: str) -> pd.DataFrame:
        path = Path(filepath).resolve()
        with open(path, encoding="utf-8") as f:
            messages: list[dict] = json.load(f)

        rows: list[dict] = [
            {
                "record_id": str(uuid.uuid4()),
                "source_format": self.source_format,
                "source_file": str(path),
                "timestamp": self._parse_ts(msg.get("timestamp", "")),
                "text": msg.get("text", ""),
                "raw_metadata": {
                    "channel": msg.get("channel", ""),
                    "username": msg.get("username", ""),
                    "user_id": msg.get("user_id", ""),
                },
            }
            for msg in messages
        ]

        df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
        log.info("SlackAdapter: loaded %d records from '%s'", len(df), path.name)
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        failures: list[str] = []

        null_ids = df["record_id"].isna() | (df["record_id"] == "")
        if null_ids.any():
            failures.append(f"record_id: {int(null_ids.sum())} null/empty value(s)")

        if df["text"].isna().any():
            failures.append(f"text: {int(df['text'].isna().sum())} null value(s)")

        if df["timestamp"].isna().any():
            failures.append(f"timestamp: {int(df['timestamp'].isna().sum())} null value(s)")

        for msg in failures:
            log.error("SlackAdapter schema violation — %s", msg)
        return not bool(failures)

    @staticmethod
    def _parse_ts(raw: str) -> datetime | None:
        if not raw:
            return None
        try:
            # datetime.fromisoformat doesn't accept a trailing "Z" before Python 3.11
            return datetime.fromisoformat(raw.rstrip("Z")).replace(tzinfo=timezone.utc)
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# CSVAdapter
# ---------------------------------------------------------------------------

class CSVAdapter(DataAdapter):
    """Ingests CSV project records with deduplication and missing-ID repair.

    Unified schema mapping
    ----------------------
    record_id     : from record_id column (or generated UUID when blank)
    source_format : "csv"
    source_file   : resolved filepath
    timestamp     : parsed from due_date column
    text          : f"{assignee_name} {notes}".strip()
    raw_metadata  : all original CSV columns + "record_id_was_missing" flag
    """

    source_format: ClassVar[str] = "csv"

    def __init__(self) -> None:
        self.duplicates_dropped: int = 0
        self.missing_ids_repaired: int = 0

    def load(self, filepath: str) -> pd.DataFrame:
        path = Path(filepath).resolve()
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        raw = raw.fillna("")

        # ── 1. Repair blank record_ids ───────────────────────────────────
        missing_mask: pd.Series = raw["record_id"].str.strip() == ""
        self.missing_ids_repaired = int(missing_mask.sum())
        new_uuids = [str(uuid.uuid4()) for _ in range(self.missing_ids_repaired)]
        raw.loc[missing_mask, "record_id"] = new_uuids
        # Keep track of the generated IDs so we can set the metadata flag later
        repaired_id_set: set[str] = set(new_uuids)

        # ── 2. Deduplicate (keep first occurrence) ───────────────────────
        before = len(raw)
        raw = raw.drop_duplicates(subset=["record_id"], keep="first").reset_index(drop=True)
        self.duplicates_dropped = before - len(raw)
        if self.duplicates_dropped:
            log.warning(
                "CSVAdapter: dropped %d duplicate record_id row(s) from '%s'",
                self.duplicates_dropped,
                path.name,
            )

        # ── 3. Build unified rows ────────────────────────────────────────
        rows: list[dict] = []
        for _, row in raw.iterrows():
            meta: dict = row.to_dict()  # preserves all original CSV columns
            if row["record_id"] in repaired_id_set:
                meta["record_id_was_missing"] = True

            text = " ".join(
                part for part in (row.get("assignee_name", ""), row.get("notes", ""))
                if part
            ).strip()

            rows.append(
                {
                    "record_id": row["record_id"],
                    "source_format": self.source_format,
                    "source_file": str(path),
                    "timestamp": self._parse_due_date(row.get("due_date", "")),
                    "text": text,
                    "raw_metadata": meta,
                }
            )

        df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
        log.info(
            "CSVAdapter: loaded %d records from '%s' "
            "(dropped %d duplicate(s), repaired %d missing ID(s))",
            len(df),
            path.name,
            self.duplicates_dropped,
            self.missing_ids_repaired,
        )
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        failures: list[str] = []
        dupes = int(df["record_id"].duplicated().sum())
        if dupes:
            failures.append(f"record_id: {dupes} duplicate(s) remain after deduplication")
        for msg in failures:
            log.error("CSVAdapter schema violation — %s", msg)
        return not bool(failures)

    @staticmethod
    def _parse_due_date(raw: str) -> datetime | None:
        if not raw.strip():
            return None
        # Try default parsing first (handles ISO, US, and most unambiguous formats).
        # Fall back with dayfirst=True for "15-Mar-2024" style.
        for dayfirst in (False, True):
            try:
                return (
                    pd.to_datetime(raw, dayfirst=dayfirst)
                    .to_pydatetime()
                    .replace(tzinfo=timezone.utc)
                )
            except (ValueError, TypeError):
                continue
        log.warning("CSVAdapter: could not parse date %r — leaving as None", raw)
        return None


# ---------------------------------------------------------------------------
# PlainTextAdapter
# ---------------------------------------------------------------------------

class PlainTextAdapter(DataAdapter):
    """Ingests plain-text files, splitting on blank lines into comment blocks.

    Unified schema mapping
    ----------------------
    record_id     : generated UUID per block
    source_format : "plaintext"
    source_file   : resolved filepath
    timestamp     : file modification time (UTC)
    text          : the comment block
    raw_metadata  : {}
    """

    source_format: ClassVar[str] = "plaintext"

    def load(self, filepath: str) -> pd.DataFrame:
        path = Path(filepath).resolve()
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        content = path.read_text(encoding="utf-8")

        # Split on double newline (blank-line-separated blocks).
        # Fall back to splitting on single newlines when the file has no blank lines
        # (e.g. one-comment-per-line files like code_comments.txt).
        blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
        if len(blocks) <= 1:
            blocks = [b.strip() for b in content.splitlines() if b.strip()]

        rows: list[dict] = [
            {
                "record_id": str(uuid.uuid4()),
                "source_format": self.source_format,
                "source_file": str(path),
                "timestamp": mtime,
                "text": block,
                "raw_metadata": {},
            }
            for block in blocks
        ]

        df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
        log.info("PlainTextAdapter: loaded %d blocks from '%s'", len(df), path.name)
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        empty = df["text"].str.strip() == ""
        if empty.any():
            log.error(
                "PlainTextAdapter schema violation — %d empty text block(s)",
                int(empty.sum()),
            )
            return False
        return True


# ---------------------------------------------------------------------------
# IngestOrchestrator
# ---------------------------------------------------------------------------

class IngestOrchestrator:
    """Runs multiple adapters, validates schemas, and concatenates into one DataFrame."""

    def ingest_all(self, config: dict[str, type[DataAdapter]]) -> pd.DataFrame:
        """Ingest all sources defined in *config*.

        Parameters
        ----------
        config:
            Mapping of ``filepath → DataAdapter subclass``.
            Example::

                {
                    "data/raw/slack_messages.json": SlackAdapter,
                    "data/raw/project_records.csv": CSVAdapter,
                }

        Returns
        -------
        pd.DataFrame
            Unified DataFrame with a ``source_lineage_tag`` column appended.
            Returns an empty DataFrame (with correct columns) if no source loads.
        """
        frames: list[pd.DataFrame] = []
        schema_failures: list[str] = []
        total_csv_dupes: int = 0
        total_missing_repaired: int = 0

        for filepath, adapter_cls in config.items():
            adapter = adapter_cls()

            try:
                df = adapter.load(filepath)
            except Exception as exc:
                log.error("Failed to load '%s': %s", filepath, exc)
                schema_failures.append(f"{Path(filepath).name}: load error — {exc}")
                continue

            if not adapter.validate_schema(df):
                schema_failures.append(f"{Path(filepath).name}: schema validation failed")

            if isinstance(adapter, CSVAdapter):
                total_csv_dupes += adapter.duplicates_dropped
                total_missing_repaired += adapter.missing_ids_repaired

            frames.append(df)

        if not frames:
            log.error("IngestOrchestrator: no data loaded — returning empty DataFrame")
            return pd.DataFrame(columns=UNIFIED_COLUMNS + ["source_lineage_tag"])

        unified = pd.concat(frames, ignore_index=True)

        # Lineage tag per row: format::filepath::timestamp
        unified["source_lineage_tag"] = (
            unified["source_format"]
            + "::"
            + unified["source_file"]
            + "::"
            + unified["timestamp"].astype(str)
        )

        # ── Summary log ──────────────────────────────────────────────────
        divider = "-" * 62
        log.info(divider)
        log.info("IngestOrchestrator complete")
        log.info("  Total records ingested       : %d", len(unified))
        for fmt in unified["source_format"].unique():
            n = int((unified["source_format"] == fmt).sum())
            log.info("  %-28s : %d records", fmt, n)
        log.info("  CSV duplicate rows dropped   : %d", total_csv_dupes)
        log.info("  Missing record IDs repaired  : %d", total_missing_repaired)
        if schema_failures:
            log.warning("  Schema validation failures   : %d", len(schema_failures))
            for failure in schema_failures:
                log.warning("    • %s", failure)
        else:
            log.info("  Schema validation failures   : 0")
        log.info(divider)

        return unified


# ---------------------------------------------------------------------------
# CLI entry point:  python -m pipeline.ingest
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary table of the unified DataFrame."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  UNIFIED DATAFRAME SUMMARY")
    print(sep)
    print(f"  Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns : {', '.join(df.columns)}")
    print()

    for fmt in df["source_format"].unique():
        sub = df[df["source_format"] == fmt]
        print(f"  [{fmt}]  {len(sub):,} records")
        for _, row in sub.head(2).iterrows():
            rid = str(row["record_id"])[:8]
            ts = str(row["timestamp"])[:19]
            txt = str(row["text"])[:55].replace("\n", " ")
            print(f"    {rid}...  {ts}  {txt!r}")
        if len(sub) > 2:
            print(f"    ... ({len(sub) - 2} more)")
        print()

    null_ts = int(df["timestamp"].isna().sum())
    empty_text = int((df["text"].isna() | (df["text"] == "")).sum())
    print(f"  Null timestamps : {null_ts}")
    print(f"  Empty text      : {empty_text}")
    print(sep)


if __name__ == "__main__":
    _project_root = Path(__file__).resolve().parent.parent
    _raw = _project_root / "data" / "raw"

    _config: dict[str, type[DataAdapter]] = {
        str(_raw / "slack_messages.json"): SlackAdapter,
        str(_raw / "project_records.csv"): CSVAdapter,
        str(_raw / "code_comments.txt"): PlainTextAdapter,
    }

    _orchestrator = IngestOrchestrator()
    _result = _orchestrator.ingest_all(_config)
    _print_summary(_result)
