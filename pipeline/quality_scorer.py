"""
NumPy-powered quality scoring model for the PII curation pipeline.

Four dimensions are scored per record (each 0.0 – 1.0), then combined
into a weighted composite ``quality_score``:

  completeness_score         weight 0.35
  consistency_score          weight 0.20
  format_conformance_score   weight 0.15
  pii_removal_confidence_score  weight 0.30

All scoring uses vectorised pandas / NumPy operations — no row-wise
``apply`` loops.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Composite weights  (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "completeness": 0.35,
    "consistency": 0.20,
    "format_conformance": 0.15,
    "pii_removal": 0.30,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# Reasonable timestamp range for the consistency dimension
_TS_MIN = pd.Timestamp("2010-01-01", tz="UTC")
_TS_MAX = pd.Timestamp("2030-12-31", tz="UTC")

# Minimum text length to earn a full completeness score
_MIN_TEXT_LEN = 10

# Quality threshold used in the report
_HIGH_QUALITY_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# QualityScorer
# ---------------------------------------------------------------------------

class QualityScorer:
    """
    Vectorised quality scorer that operates on a unified-schema DataFrame.

    Expected columns (produced by the ingest + anonymisation pipeline):

    ``text``                  str   – record body
    ``timestamp``             datetime-like – record timestamp
    ``source_lineage_tag``    str   – "format::file::timestamp"
    ``has_pii``               bool  – from PIIDetector.detect_batch()
    ``pii_count``             int   – total PII detections
    ``anonymization_count``   int   – replacements made by Anonymizer
    """

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score every row across four quality dimensions and compute a
        composite ``quality_score``.

        Returns a **copy** of *df* with five new float columns:
        ``completeness_score``, ``consistency_score``,
        ``format_conformance_score``, ``pii_removal_confidence_score``,
        ``quality_score``.
        """
        out = df.copy()

        out["completeness_score"] = self._completeness(out)
        out["consistency_score"] = self._consistency(out)
        out["format_conformance_score"] = self._format_conformance(out)
        out["pii_removal_confidence_score"] = self._pii_removal(out)

        out["quality_score"] = np.clip(
            WEIGHTS["completeness"] * out["completeness_score"]
            + WEIGHTS["consistency"] * out["consistency_score"]
            + WEIGHTS["format_conformance"] * out["format_conformance_score"]
            + WEIGHTS["pii_removal"] * out["pii_removal_confidence_score"],
            0.0,
            1.0,
        )

        return out

    def generate_report(self, df: pd.DataFrame, output_path: str) -> dict:
        """
        Compute summary statistics over a scored DataFrame and persist them.

        *df* must already have a ``quality_score`` column (call
        :meth:`score_batch` first).  The report is written as JSON to
        *output_path* (parent directories are created automatically).

        Returns the report dict.
        """
        _require_col(df, "quality_score", "call score_batch() first")

        qs = df["quality_score"]
        n = len(df)

        # PII columns are optional — default gracefully if absent
        has_pii = df["has_pii"].astype(bool) if "has_pii" in df.columns else pd.Series([False] * n)
        pii_count = df["pii_count"] if "pii_count" in df.columns else pd.Series([0] * n)
        anon_count = df["anonymization_count"] if "anonymization_count" in df.columns else pd.Series([0] * n)

        pii_rows = int(has_pii.sum())
        fully_anonymized = int(((has_pii) & (anon_count >= pii_count) & (pii_count > 0)).sum())

        report = {
            "record_count": n,
            "mean_quality_score": round(float(qs.mean()), 4),
            "median_quality_score": round(float(qs.median()), 4),
            "std_quality_score": round(float(qs.std(ddof=1)) if n > 1 else 0.0, 4),
            "pct_above_threshold": round(
                float((qs >= _HIGH_QUALITY_THRESHOLD).sum()) / max(n, 1) * 100, 2
            ),
            "threshold_used": _HIGH_QUALITY_THRESHOLD,
            "pct_records_with_pii": round(pii_rows / max(n, 1) * 100, 2),
            "pct_pii_records_fully_anonymized": round(
                fully_anonymized / max(pii_rows, 1) * 100, 2
            ),
            "weights": WEIGHTS,
        }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return report

    # ------------------------------------------------------------------
    # Dimension 1 — Completeness
    # ------------------------------------------------------------------

    def _completeness(self, df: pd.DataFrame) -> pd.Series:
        """
        Score based on the ``text`` column length.

        1.0  text non-null and len > 10
        0.5  text non-null but len ≤ 10
        0.0  text null or empty
        """
        if "text" not in df.columns:
            return pd.Series(np.zeros(len(df)), index=df.index)

        text = df["text"].fillna("")
        length = text.str.len()

        return pd.Series(
            np.select(
                [length > _MIN_TEXT_LEN, length > 0],
                [1.0,                    0.5],
                default=0.0,
            ),
            index=df.index,
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Dimension 2 — Consistency
    # ------------------------------------------------------------------

    def _consistency(self, df: pd.DataFrame) -> pd.Series:
        """
        Score based on whether ``timestamp`` is parseable and in [2010, 2030].

        1.0  parseable and within range
        0.5  present but unparseable or out of range
        0.0  missing / null
        """
        if "timestamp" not in df.columns:
            return pd.Series(np.zeros(len(df)), index=df.index)

        raw = df["timestamp"]
        missing = raw.isna()

        # Coerce to datetime; errors become NaT
        parsed = pd.to_datetime(raw, errors="coerce", utc=True)
        parseable = parsed.notna()
        in_range = parseable & (parsed >= _TS_MIN) & (parsed <= _TS_MAX)

        return pd.Series(
            np.select(
                [missing,  in_range, parseable],
                [0.0,      1.0,      0.5],
                default=0.0,
            ),
            index=df.index,
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Dimension 3 — Format conformance
    # ------------------------------------------------------------------

    def _format_conformance(self, df: pd.DataFrame) -> pd.Series:
        """
        Score based on whether ``source_lineage_tag`` matches
        the expected ``format::file::timestamp`` pattern.

        1.0  tag present and contains at least two "::" separators
        0.5  tag present but only one "::" separator (partially formed)
        0.0  tag missing / null / empty
        """
        if "source_lineage_tag" not in df.columns:
            return pd.Series(np.zeros(len(df)), index=df.index)

        tag = df["source_lineage_tag"].fillna("")
        missing = tag == ""

        # Count occurrences of "::" in each tag via vectorised string ops
        sep_count = tag.str.count(r"::")

        return pd.Series(
            np.select(
                [missing,  sep_count >= 2, sep_count == 1],
                [0.0,      1.0,            0.5],
                default=0.0,
            ),
            index=df.index,
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Dimension 4 — PII removal confidence
    # ------------------------------------------------------------------

    def _pii_removal(self, df: pd.DataFrame) -> pd.Series:
        """
        Score based on PII detection and anonymisation columns.

        1.0   has_pii is False  (no PII present)
        0.95  has_pii True  AND  anonymization_count >= pii_count  (fully anonymized)
        0.5   has_pii True  AND  0 < anonymization_count < pii_count  (partial)
        0.0   has_pii True  AND  anonymization_count == 0  (not anonymized)
        """
        n = len(df)

        # Graceful defaults when columns are absent
        has_pii = (
            df["has_pii"].astype(bool)
            if "has_pii" in df.columns
            else pd.Series([False] * n, index=df.index)
        )
        pii_count = (
            df["pii_count"].fillna(0).astype(int)
            if "pii_count" in df.columns
            else pd.Series([0] * n, index=df.index)
        )
        anon_count = (
            df["anonymization_count"].fillna(0).astype(int)
            if "anonymization_count" in df.columns
            else pd.Series([0] * n, index=df.index)
        )

        no_pii = ~has_pii
        fully_anon = has_pii & (anon_count >= pii_count) & (pii_count > 0)
        partial_anon = has_pii & (anon_count > 0) & (anon_count < pii_count)
        not_anon = has_pii & (anon_count == 0)

        return pd.Series(
            np.select(
                [no_pii, fully_anon, partial_anon, not_anon],
                [1.0,    0.95,       0.5,          0.0],
                default=0.0,
            ),
            index=df.index,
            dtype=float,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_col(df: pd.DataFrame, col: str, hint: str = "") -> None:
    if col not in df.columns:
        msg = f"DataFrame is missing required column '{col}'."
        if hint:
            msg += f" Hint: {hint}"
        raise ValueError(msg)
