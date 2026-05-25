"""
Anonymization engine for the PII curation pipeline.

Two strategies are applied based on PII type:

Pseudonymization (hash-based, referentially stable)
    Types : EMAIL, PHONE_US, SSN, CREDIT_CARD, PERSON
    Token : [TYPE_XXXXXXXX]  where XXXXXXXX = sha256(matched_text)[:8]
    Example: "john@example.com"  →  "[EMAIL_a94f2c1b]"
    The same input always produces the same token, so referential
    integrity is preserved when the same value appears in multiple records.

Redaction (type-only placeholder, no information retained)
    Types : IP_ADDRESS, ORG, GPE
    Token : [REDACTED_TYPE]
    Example: "Google LLC"  →  "[REDACTED_ORG]"
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from pipeline.pii_detector import DetectionDict

# ---------------------------------------------------------------------------
# Strategy routing tables
# ---------------------------------------------------------------------------

PSEUDONYMIZE_TYPES: frozenset[str] = frozenset(
    {"EMAIL", "PHONE_US", "SSN", "CREDIT_CARD", "PERSON"}
)

REDACT_TYPES: frozenset[str] = frozenset({"IP_ADDRESS", "ORG", "GPE"})

# Convenience alias for callers that need to inspect log entries
LogEntry = dict[str, Any]
# Keys guaranteed in every LogEntry:
#   original_text : str
#   replacement   : str
#   pii_type      : str
#   strategy      : "pseudonymization" | "redaction"
#   position      : {"start": int, "end": int}


# ---------------------------------------------------------------------------
# Token constructors  (module-level so tests can call them directly)
# ---------------------------------------------------------------------------

def make_pseudo_token(pii_type: str, matched_text: str) -> str:
    """Return ``[TYPE_XXXXXXXX]`` where XXXXXXXX is the first 8 hex chars of SHA-256.

    Deterministic: same (pii_type, matched_text) pair always yields the same token.
    """
    digest = hashlib.sha256(matched_text.encode("utf-8")).hexdigest()[:8]
    return f"[{pii_type}_{digest}]"


def make_redact_token(pii_type: str) -> str:
    """Return ``[REDACTED_TYPE]``."""
    return f"[REDACTED_{pii_type}]"


# ---------------------------------------------------------------------------
# Anonymizer
# ---------------------------------------------------------------------------

class Anonymizer:
    """
    Apply pseudonymization or redaction to pre-detected PII spans.

    The class is stateless — no caches, no randomness.  Pass the
    ``detections`` list returned by :meth:`~pipeline.pii_detector.PIIDetector.detect`
    directly into :meth:`anonymize`.
    """

    # ------------------------------------------------------------------
    # Primary public methods
    # ------------------------------------------------------------------

    def anonymize(
        self,
        text: str,
        detections: list[DetectionDict],
    ) -> tuple[str, list[LogEntry]]:
        """
        Replace every detected PII span with the appropriate token.

        Parameters
        ----------
        text:
            The original text string.
        detections:
            Detection dicts from :meth:`PIIDetector.detect`.
            Assumed already deduplicated; still sorted defensively before
            processing to guarantee correct character-offset arithmetic.

        Returns
        -------
        anonymized_text : str
            Text with all PII spans replaced.
        anonymization_log : list[LogEntry]
            One entry per replacement, sorted by original start position.
            Each entry contains:
            ``original_text``, ``replacement``, ``pii_type``,
            ``strategy``, ``position`` (``{"start": int, "end": int}``).
        """
        if not detections:
            return text, []

        # Defensive sort by start position, then reverse for right-to-left
        # replacement so earlier offsets are never invalidated by substitutions.
        ordered: list[DetectionDict] = sorted(
            detections, key=lambda d: d["start"], reverse=True
        )

        log: list[LogEntry] = []
        for det in ordered:
            pii_type: str = det["pii_type"]
            matched: str = det["matched_text"]
            start: int = det["start"]
            end: int = det["end"]

            replacement, strategy = self._resolve(pii_type, matched)
            text = text[:start] + replacement + text[end:]

            log.append(
                {
                    "original_text": matched,
                    "replacement": replacement,
                    "pii_type": pii_type,
                    "strategy": strategy,
                    "position": {"start": start, "end": end},
                }
            )

        # Return log in natural (left-to-right) reading order
        log.sort(key=lambda e: e["position"]["start"])
        return text, log

    def anonymize_batch(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
    ) -> pd.DataFrame:
        """
        Anonymize every row of *df*, which must already have a
        ``pii_detections`` column from :meth:`PIIDetector.detect_batch`.

        Parameters
        ----------
        df:
            DataFrame with at least ``pii_detections`` and *text_col* columns.
        text_col:
            Name of the column containing original text.  Defaults to
            ``"text"`` (the unified schema column name from the ingest pipeline).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with four new columns:

            ``anonymized_text``
                str — text after anonymization.
            ``anonymization_log``
                list[LogEntry] — one entry per replacement.
            ``anonymization_count``
                int — number of PII spans replaced in this row.
            ``strategies_used``
                set[str] — which strategies were applied (subset of
                ``{"pseudonymization", "redaction"}``).
        """
        if "pii_detections" not in df.columns:
            raise ValueError(
                "DataFrame is missing the 'pii_detections' column. "
                "Call PIIDetector.detect_batch() before anonymize_batch()."
            )
        if text_col not in df.columns:
            raise ValueError(
                f"DataFrame is missing the text column '{text_col}'. "
                f"Available columns: {list(df.columns)}"
            )

        out = df.copy()
        anon_texts: list[str] = []
        anon_logs: list[list[LogEntry]] = []

        for text, detections in zip(
            out[text_col].fillna("").astype(str),
            out["pii_detections"],
        ):
            anon_text, log = self.anonymize(text, detections if isinstance(detections, list) else [])
            anon_texts.append(anon_text)
            anon_logs.append(log)

        out["anonymized_text"] = anon_texts
        out["anonymization_log"] = anon_logs
        out["anonymization_count"] = out["anonymization_log"].apply(len)
        out["strategies_used"] = out["anonymization_log"].apply(
            lambda entries: {e["strategy"] for e in entries}
        )
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, pii_type: str, matched_text: str) -> tuple[str, str]:
        """Return ``(replacement_token, strategy_name)`` for the given PII type."""
        if pii_type in PSEUDONYMIZE_TYPES:
            return make_pseudo_token(pii_type, matched_text), "pseudonymization"
        if pii_type in REDACT_TYPES:
            return make_redact_token(pii_type), "redaction"
        # Unknown type: fall back to generic redaction to avoid leaking data
        return f"[REDACTED_{pii_type}]", "redaction"
