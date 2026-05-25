"""
Structured data lineage logging system for the PII curation pipeline.

Events are buffered in memory and flushed to a newline-delimited JSON
(JSONL) file — one JSON object per line — so the log is human-readable,
grep-able, and appendable without loading the entire file into memory.

A compact summary sidecar is written alongside the JSONL on every flush.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _make_run_id() -> str:
    """
    Create a timestamp-based run identifier.

    Format: ``YYYYMMDD_HHMMSS_<8-hex-chars>``
    Example: ``20240615_143022_a3f8c1d2``

    The timestamp prefix makes log files sort chronologically;
    the UUID suffix guarantees uniqueness when multiple runs start
    within the same second.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    return f"{ts}_{uid}"


def _to_json_safe(value: Any) -> Any:
    """
    Convert non-JSON-serializable types to safe equivalents.

    datetime  → ISO-8601 string
    set/frozenset → sorted list
    Path      → str
    Everything else is returned as-is (assumed already serializable).
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# LineageLogger
# ---------------------------------------------------------------------------

class LineageLogger:
    """
    Record-level lineage logger for the PII curation pipeline.

    One logger instance covers a single pipeline run.  Call the
    appropriate ``log_*`` method after each stage, then call
    :meth:`flush` once at the end of the run to persist all buffered
    events and generate a summary sidecar.

    Parameters
    ----------
    output_dir:
        Directory where JSONL and summary files will be written.
        Created automatically if it does not exist.
    run_id:
        Optional stable identifier for this run.  A timestamp-based
        UUID is generated automatically when *run_id* is ``None``.
    """

    def __init__(self, output_dir: str, run_id: str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id: str = run_id if run_id is not None else _make_run_id()
        self._events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Stage loggers
    # ------------------------------------------------------------------

    def log_ingestion(
        self,
        record_id: str,
        source_format: str,
        source_file: str,
        timestamp: Any,
        text_length: int,
    ) -> None:
        """
        Log a record-ingestion event.

        Parameters
        ----------
        record_id:     Unique identifier for the record.
        source_format: Adapter type, e.g. "slack", "csv", "plaintext".
        source_file:   Path of the originating file.
        timestamp:     Source-level timestamp (datetime or ISO string).
        text_length:   Character length of the record's ``text`` field.
        """
        self._append(
            {
                "event": "ingested",
                "run_id": self.run_id,
                "record_id": record_id,
                "source_format": source_format,
                "source_file": source_file,
                "ingestion_timestamp": _now_iso(),
                "source_timestamp": _to_json_safe(timestamp),
                "text_length": text_length,
            }
        )

    def log_pii_detection(
        self,
        record_id: str,
        pii_types_found: list[str] | set[str],
        pii_count: int,
        detection_timestamp: Any,
    ) -> None:
        """
        Log a PII-detection event for a single record.

        Parameters
        ----------
        record_id:           Record this detection belongs to.
        pii_types_found:     Collection of PII-type strings found, e.g.
                             {"EMAIL", "PERSON"}.  Sets are sorted before
                             serialisation for reproducibility.
        pii_count:           Total number of individual PII spans detected.
        detection_timestamp: When detection ran (datetime or ISO string).
        """
        self._append(
            {
                "event": "pii_detected",
                "run_id": self.run_id,
                "record_id": record_id,
                "pii_types_found": sorted(pii_types_found),
                "pii_count": pii_count,
                "detection_timestamp": _to_json_safe(detection_timestamp),
            }
        )

    def log_anonymization(
        self,
        record_id: str,
        strategy: str,
        pii_type: str,
        original_length: int,
        replacement: str,
        anonymization_timestamp: Any,
    ) -> None:
        """
        Log one anonymization action.

        Call once per PII span replaced (not once per record) so the log
        captures every individual substitution for full auditability.

        Parameters
        ----------
        record_id:               Record the action was applied to.
        strategy:                "pseudonymization" or "redaction".
        pii_type:                PII type replaced, e.g. "EMAIL".
        original_length:         Character length of the original PII span.
        replacement:             The token written in place of the PII span.
        anonymization_timestamp: When anonymization ran.
        """
        self._append(
            {
                "event": "anonymized",
                "run_id": self.run_id,
                "record_id": record_id,
                "strategy": strategy,
                "pii_type": pii_type,
                "original_length": original_length,
                "replacement": replacement,
                "anonymization_timestamp": _to_json_safe(anonymization_timestamp),
            }
        )

    def log_quality_score(
        self,
        record_id: str,
        quality_score: float,
        completeness_score: float,
        pii_removal_confidence_score: float,
        passed_threshold: bool,
    ) -> None:
        """
        Log a quality-scoring event for a single record.

        Parameters
        ----------
        record_id:                    Record that was scored.
        quality_score:                Composite score in [0, 1].
        completeness_score:           Completeness dimension score.
        pii_removal_confidence_score: PII-removal dimension score.
        passed_threshold:             True when quality_score >= the
                                      configured passing threshold.
        """
        self._append(
            {
                "event": "quality_scored",
                "run_id": self.run_id,
                "record_id": record_id,
                "quality_score": quality_score,
                "completeness_score": completeness_score,
                "pii_removal_confidence_score": pii_removal_confidence_score,
                "passed_threshold": passed_threshold,
            }
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self, filename: str = "lineage_log.jsonl") -> str:
        """
        Persist all buffered events and generate a summary sidecar.

        The JSONL file contains one JSON object per line — no surrounding
        array — so it is grep-able and appendable without parsing the
        whole file.  The summary sidecar is written to the same directory
        with ``_summary.json`` appended to the stem.

        The in-memory buffer is cleared after writing.

        Parameters
        ----------
        filename:
            Name of the JSONL output file (relative to *output_dir*).

        Returns
        -------
        str
            Absolute path to the JSONL lineage log file.
        """
        log_path = self.output_dir / filename
        stem = Path(filename).stem
        summary_path = self.output_dir / f"{stem}_summary.json"

        # ── JSONL: one compact JSON object per line ───────────────────
        with open(log_path, "w", encoding="utf-8") as fh:
            for event in self._events:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")

        # ── Summary sidecar ───────────────────────────────────────────
        type_counts: Counter[str] = Counter(e["event"] for e in self._events)
        unique_records: int = len({e["record_id"] for e in self._events})

        summary: dict[str, Any] = {
            "run_id": self.run_id,
            "total_events": len(self._events),
            "events_by_type": dict(type_counts),
            "records_processed": unique_records,
            "log_file": str(log_path.resolve()),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._events.clear()
        return str(log_path.resolve())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append(self, event: dict[str, Any]) -> None:
        """Add *event* to the in-memory buffer."""
        self._events.append(event)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def event_count(self) -> int:
        """Number of events currently buffered (unflushed)."""
        return len(self._events)

    def __len__(self) -> int:
        return self.event_count

    def __repr__(self) -> str:
        return (
            f"LineageLogger(run_id={self.run_id!r}, "
            f"output_dir={str(self.output_dir)!r}, "
            f"buffered_events={self.event_count})"
        )
