"""
Two-layer PII detection system.

Layer 1 – compiled regex patterns for structured identifiers:
    EMAIL, PHONE_US, SSN, CREDIT_CARD, IP_ADDRESS

Layer 2 – spaCy named-entity recognition for contextual entities:
    PERSON, ORG, GPE

Run:
    python -m spacy download en_core_web_sm
before importing this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from re import Pattern
from typing import Any

import pandas as pd
import spacy
from spacy.language import Language


# ---------------------------------------------------------------------------
# PIIPattern – config dataclass for each regex rule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PIIPattern:
    name: str
    pattern: Pattern[str]
    pii_type: str
    confidence: float


# ---------------------------------------------------------------------------
# Layer 1 – Regex patterns
# ---------------------------------------------------------------------------

REGEX_PATTERNS: list[PIIPattern] = [
    PIIPattern(
        name="EMAIL",
        pattern=re.compile(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
            re.IGNORECASE,
        ),
        pii_type="EMAIL",
        confidence=0.99,
    ),
    PIIPattern(
        name="PHONE_US",
        pattern=re.compile(
            r"""(?x)
            (?:
                # Formatted: (xxx) xxx-xxxx  or  xxx-xxx-xxxx  or  xxx.xxx.xxxx
                (?:\+?1[\s.\-]?)?                    # optional  +1  country code
                (?:
                    \(\d{3}\)[\s.\-]?                # (xxx)  area code
                  | \d{3}[\s.\-]                     # xxx-  (separator required to
                )                                    #        avoid bare-digit false hits)
                \d{3}[\s.\-]?\d{4}
              | \b\d{10}\b                           # bare 10-digit: xxxxxxxxxx
            )
            """,
        ),
        pii_type="PHONE_US",
        confidence=0.95,
    ),
    PIIPattern(
        name="SSN",
        # Requires word boundaries; xxx-xx-xxxx only (no spaces or dots)
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        pii_type="SSN",
        confidence=0.99,
    ),
    PIIPattern(
        name="CREDIT_CARD",
        # 13-16 digits in groups of 4 with optional dash/space separators.
        # Luhn validation is intentionally omitted here; callers that need it
        # can filter the returned detections with a Luhn check post-hoc.
        pattern=re.compile(r"\b(?:\d{4}[-\s]?){3}\d{1,4}\b"),
        pii_type="CREDIT_CARD",
        confidence=0.95,
    ),
    PIIPattern(
        name="IP_ADDRESS",
        # Strict IPv4: each octet is 0-255
        pattern=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        pii_type="IP_ADDRESS",
        confidence=0.85,
    ),
]

# Flat dict of {pii_type: compiled Pattern} – kept for backwards compatibility
# and for use in tests / anonymizer.
PATTERNS: dict[str, Pattern[str]] = {p.pii_type: p.pattern for p in REGEX_PATTERNS}


# ---------------------------------------------------------------------------
# Layer 2 – spaCy NER
# ---------------------------------------------------------------------------

# Confidence assigned per spaCy entity label
NER_CONFIDENCE: dict[str, float] = {
    "PERSON": 0.80,
    "ORG": 0.70,
    "GPE": 0.65,
}

_NER_LABELS: frozenset[str] = frozenset(NER_CONFIDENCE)


def _load_nlp(model: str = "en_core_web_sm") -> Language:
    """Load spaCy model, raising a clear OSError when it is not installed."""
    try:
        return spacy.load(model)
    except OSError:
        raise OSError(
            f"spaCy model '{model}' not found. "
            "Run: python -m spacy download en_core_web_sm"
        ) from None


# ---------------------------------------------------------------------------
# Detection dict type alias
# ---------------------------------------------------------------------------

# Each detection returned by detect() has exactly these keys:
#   pii_type     : str   – e.g. "EMAIL", "PERSON"
#   matched_text : str   – the matched substring
#   start        : int   – character offset (inclusive)
#   end          : int   – character offset (exclusive)
#   confidence   : float – in (0, 1]
#   layer        : str   – "regex" | "ner"
DetectionDict = dict[str, Any]


# ---------------------------------------------------------------------------
# PIIDetector
# ---------------------------------------------------------------------------

class PIIDetector:
    """
    Combines regex and spaCy NER to surface PII spans in free text.

    Overlapping detections are deduplicated: the highest-confidence match
    is kept.  When confidence is tied, regex is preferred over NER.
    The final list is sorted by ``start`` position.
    """

    def __init__(self, model: str = "en_core_web_sm") -> None:
        # _load_nlp raises OSError with a clear message if the model is absent.
        self.nlp: Language = _load_nlp(model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> list[DetectionDict]:
        """
        Detect PII in *text* using both regex and NER layers.

        Returns
        -------
        list[DetectionDict]
            Deduplicated, start-sorted list of detections.  Each dict has:
            ``pii_type``, ``matched_text``, ``start``, ``end``,
            ``confidence``, ``layer``.
        """
        if not text:
            return []
        raw: list[DetectionDict] = self._regex_detections(text) + self._ner_detections(text)
        return self._deduplicate(raw)

    def detect_batch(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Run :meth:`detect` over every row in *df[text_col]*.

        Returns a copy of *df* with five new columns appended:

        ``pii_detections``
            ``list[DetectionDict]`` – one list per row.
        ``pii_types_found``
            ``set[str]`` – distinct ``pii_type`` values found in the row.
        ``has_pii``
            ``bool`` – True when at least one detection was found.
        ``pii_count``
            ``int`` – total number of detections in the row.
        ``max_pii_confidence``
            ``float`` – highest confidence among detections; 0.0 if none.
        """
        out = df.copy()
        all_detections: list[list[DetectionDict]] = [
            self.detect(str(t)) for t in out[text_col].fillna("")
        ]

        out["pii_detections"] = all_detections
        out["pii_types_found"] = out["pii_detections"].apply(
            lambda ds: {d["pii_type"] for d in ds}
        )
        out["has_pii"] = out["pii_detections"].apply(bool)
        out["pii_count"] = out["pii_detections"].apply(len)
        out["max_pii_confidence"] = out["pii_detections"].apply(
            lambda ds: max((d["confidence"] for d in ds), default=0.0)
        )
        return out

    # ------------------------------------------------------------------
    # Layer 1 – regex
    # ------------------------------------------------------------------

    def _regex_detections(self, text: str) -> list[DetectionDict]:
        results: list[DetectionDict] = []
        for pii_pat in REGEX_PATTERNS:
            for m in pii_pat.pattern.finditer(text):
                results.append(
                    {
                        "pii_type": pii_pat.pii_type,
                        "matched_text": m.group(),
                        "start": m.start(),
                        "end": m.end(),
                        "confidence": pii_pat.confidence,
                        "layer": "regex",
                    }
                )
        return results

    # ------------------------------------------------------------------
    # Layer 2 – NER
    # ------------------------------------------------------------------

    def _ner_detections(self, text: str) -> list[DetectionDict]:
        doc = self.nlp(text)
        results: list[DetectionDict] = []
        for ent in doc.ents:
            conf = NER_CONFIDENCE.get(ent.label_)
            if conf is None:
                continue  # entity type not in our target set
            results.append(
                {
                    "pii_type": ent.label_,
                    "matched_text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": conf,
                    "layer": "ner",
                }
            )
        return results

    # ------------------------------------------------------------------
    # Overlap deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, detections: list[DetectionDict]) -> list[DetectionDict]:
        """
        Remove overlapping spans, keeping the highest-confidence detection.
        Regex wins over NER at equal confidence.

        Strategy: sort by (confidence DESC, layer priority, start) so the
        "best" candidate in any overlapping group is processed first.  Then
        greedily accept each detection only if it does not overlap with any
        already-kept detection.
        """
        if not detections:
            return []

        _layer_rank: dict[str, int] = {"regex": 0, "ner": 1}
        ordered = sorted(
            detections,
            key=lambda d: (-d["confidence"], _layer_rank[d["layer"]], d["start"]),
        )

        kept: list[DetectionDict] = []
        for det in ordered:
            if not any(
                det["start"] < k["end"] and k["start"] < det["end"]
                for k in kept
            ):
                kept.append(det)

        return sorted(kept, key=lambda d: d["start"])
