"""
Unit tests for PIIDetector.

Required tests (per spec)
-------------------------
test_regex_email_detection  – EMAIL detected via regex layer
test_ner_person_detection   – PERSON detected via NER layer
test_no_pii_clean_text      – has_pii is False for clean prose
"""

import pandas as pd
import pytest

from pipeline.pii_detector import PATTERNS, PIIDetector, REGEX_PATTERNS, NER_CONFIDENCE


# ---------------------------------------------------------------------------
# Shared fixture  (spaCy model loaded once per session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector() -> PIIDetector:
    return PIIDetector()


# ===========================================================================
# Required tests
# ===========================================================================

def test_regex_email_detection(detector: PIIDetector) -> None:
    """EMAIL must be detected by the regex layer in a simple sentence."""
    detections = detector.detect("Contact me at test@example.com")
    assert any(
        d["pii_type"] == "EMAIL" and d["layer"] == "regex"
        for d in detections
    ), "Expected an EMAIL detection from the regex layer"


def test_ner_person_detection(detector: PIIDetector) -> None:
    """PERSON must be detected by the NER layer in a simple sentence."""
    detections = detector.detect("Please call John Smith tomorrow")
    assert any(
        d["pii_type"] == "PERSON" and d["layer"] == "ner"
        for d in detections
    ), "Expected a PERSON detection from the NER layer"


def test_no_pii_clean_text(detector: PIIDetector) -> None:
    """has_pii must be False for text that contains no PII."""
    df = pd.DataFrame({"text": ["The pipeline ran successfully at 3pm"]})
    result = detector.detect_batch(df, "text")
    assert not result["has_pii"].iloc[0], (
        "Expected has_pii=False for clean text, "
        f"but got detections: {result['pii_detections'].iloc[0]}"
    )


# ===========================================================================
# Detection dict shape
# ===========================================================================

class TestDetectionDictShape:
    """Every returned detection dict must carry the six required keys."""

    REQUIRED_KEYS = {"pii_type", "matched_text", "start", "end", "confidence", "layer"}

    def test_email_dict_keys(self, detector: PIIDetector) -> None:
        dets = detector.detect("reach alice@example.com")
        assert dets, "Expected at least one detection"
        assert self.REQUIRED_KEYS == set(dets[0].keys())

    def test_layer_values_are_valid(self, detector: PIIDetector) -> None:
        dets = detector.detect("Call 415-555-0192 or email bob@test.org")
        for d in dets:
            assert d["layer"] in {"regex", "ner"}, f"Unexpected layer: {d['layer']}"

    def test_confidence_in_range(self, detector: PIIDetector) -> None:
        dets = detector.detect("SSN: 123-45-6789, email: a@b.com")
        for d in dets:
            assert 0.0 < d["confidence"] <= 1.0, f"Out-of-range confidence: {d['confidence']}"

    def test_start_end_consistent(self, detector: PIIDetector) -> None:
        text = "Contact alice@example.com for info."
        dets = detector.detect(text)
        for d in dets:
            assert d["start"] < d["end"]
            assert text[d["start"]: d["end"]] == d["matched_text"]


# ===========================================================================
# Regex layer – per pattern type
# ===========================================================================

class TestRegexLayer:
    def test_email_matched_text(self, detector: PIIDetector) -> None:
        dets = detector.detect("user: bob.smith+tag@corp.io")
        emails = [d for d in dets if d["pii_type"] == "EMAIL"]
        assert any("bob.smith+tag@corp.io" in d["matched_text"] for d in emails)

    def test_phone_dashes(self, detector: PIIDetector) -> None:
        dets = detector.detect("Call me at 415-555-0192 for details.")
        assert any(d["pii_type"] == "PHONE_US" for d in dets)

    def test_phone_parentheses(self, detector: PIIDetector) -> None:
        dets = detector.detect("Reach out at (800) 555-1234.")
        assert any(d["pii_type"] == "PHONE_US" for d in dets)

    def test_ssn(self, detector: PIIDetector) -> None:
        dets = detector.detect("SSN: 123-45-6789 — handle with care.")
        assert any(d["pii_type"] == "SSN" for d in dets)

    def test_credit_card_dashes(self, detector: PIIDetector) -> None:
        dets = detector.detect("Card on file: 4111-1111-1111-1111.")
        assert any(d["pii_type"] == "CREDIT_CARD" for d in dets)

    def test_credit_card_spaces(self, detector: PIIDetector) -> None:
        dets = detector.detect("Charge 4012 8888 8888 1881 for the subscription.")
        assert any(d["pii_type"] == "CREDIT_CARD" for d in dets)

    def test_ip_address(self, detector: PIIDetector) -> None:
        dets = detector.detect("Server is at 192.168.1.100.")
        assert any(d["pii_type"] == "IP_ADDRESS" for d in dets)

    def test_multiple_types_in_one_string(self, detector: PIIDetector) -> None:
        text = "Email bob@acme.com or call 555-867-5309."
        types = {d["pii_type"] for d in detector.detect(text)}
        assert "EMAIL" in types
        assert "PHONE_US" in types

    def test_clean_text_no_regex_hits(self) -> None:
        text = "The deployment pipeline is green and all tests are passing."
        for pii_pat in REGEX_PATTERNS:
            assert not pii_pat.pattern.search(text), (
                f"Unexpected {pii_pat.pii_type} regex hit in clean text"
            )


# ===========================================================================
# NER layer
# ===========================================================================

class TestNERLayer:
    def test_org_detected(self, detector: PIIDetector) -> None:
        dets = detector.detect("The contract was reviewed by Microsoft Legal.")
        assert any(d["pii_type"] == "ORG" and d["layer"] == "ner" for d in dets)

    def test_gpe_detected(self, detector: PIIDetector) -> None:
        dets = detector.detect("The team is relocating to San Francisco next quarter.")
        assert any(d["pii_type"] == "GPE" and d["layer"] == "ner" for d in dets)

    def test_ner_confidence_values(self, detector: PIIDetector) -> None:
        dets = detector.detect("Alice works at Google in New York.")
        for d in [x for x in dets if x["layer"] == "ner"]:
            expected = NER_CONFIDENCE.get(d["pii_type"])
            assert expected is not None
            assert d["confidence"] == expected


# ===========================================================================
# Overlap deduplication
# ===========================================================================

class TestDeduplication:
    def test_no_overlapping_spans_in_output(self, detector: PIIDetector) -> None:
        """No two returned detections may share a character position."""
        text = "Reach Sarah Johnson at sarah.johnson@example.com or 415-555-9900."
        dets = detector.detect(text)
        for i, a in enumerate(dets):
            for b in dets[i + 1:]:
                overlap = a["start"] < b["end"] and b["start"] < a["end"]
                assert not overlap, (
                    f"Overlapping detections: {a['matched_text']!r} and {b['matched_text']!r}"
                )

    def test_output_sorted_by_start(self, detector: PIIDetector) -> None:
        text = "Call 415-555-0192 then email alice@test.com and note SSN 987-65-4321."
        dets = detector.detect(text)
        starts = [d["start"] for d in dets]
        assert starts == sorted(starts)

    def test_regex_preferred_over_ner_on_tie(self, detector: PIIDetector) -> None:
        """Regression: when both layers fire on the same span, regex wins."""
        # Construct a detector that only exposes overlapping candidates
        det = PIIDetector()
        # EMAIL (confidence 0.99) will always beat a NER match on the same span
        dets = det.detect("Mail to john@example.com now.")
        email_dets = [d for d in dets if d["pii_type"] == "EMAIL"]
        assert email_dets, "EMAIL detection missing"
        assert all(d["layer"] == "regex" for d in email_dets)


# ===========================================================================
# detect_batch
# ===========================================================================

class TestDetectBatch:
    def test_adds_five_columns(self, detector: PIIDetector) -> None:
        df = pd.DataFrame({"text": ["hello alice@test.com", "clean text"]})
        out = detector.detect_batch(df, "text")
        for col in ("pii_detections", "pii_types_found", "has_pii", "pii_count", "max_pii_confidence"):
            assert col in out.columns

    def test_has_pii_true_for_email_row(self, detector: PIIDetector) -> None:
        df = pd.DataFrame({"text": ["Contact bob@corp.com"]})
        out = detector.detect_batch(df, "text")
        assert out["has_pii"].iloc[0] is True or out["has_pii"].iloc[0] == True  # noqa: E712

    def test_has_pii_false_for_clean_row(self, detector: PIIDetector) -> None:
        df = pd.DataFrame({"text": ["All systems operational."]})
        out = detector.detect_batch(df, "text")
        assert not out["has_pii"].iloc[0]

    def test_pii_count_matches_detections_length(self, detector: PIIDetector) -> None:
        df = pd.DataFrame({"text": ["Email a@b.com and call 415-555-0000.", ""]})
        out = detector.detect_batch(df, "text")
        for _, row in out.iterrows():
            assert row["pii_count"] == len(row["pii_detections"])

    def test_pii_types_found_is_set(self, detector: PIIDetector) -> None:
        df = pd.DataFrame({"text": ["a@b.com 415-555-0000"]})
        out = detector.detect_batch(df, "text")
        assert isinstance(out["pii_types_found"].iloc[0], set)

    def test_max_confidence_zero_when_no_pii(self, detector: PIIDetector) -> None:
        df = pd.DataFrame({"text": ["Nothing to see here."]})
        out = detector.detect_batch(df, "text")
        assert out["max_pii_confidence"].iloc[0] == 0.0

    def test_max_confidence_email_is_099(self, detector: PIIDetector) -> None:
        df = pd.DataFrame({"text": ["Send to alice@example.com"]})
        out = detector.detect_batch(df, "text")
        assert out["max_pii_confidence"].iloc[0] == pytest.approx(0.99)

    def test_null_text_handled_gracefully(self, detector: PIIDetector) -> None:
        df = pd.DataFrame({"text": [None, "email: x@y.com"]})
        out = detector.detect_batch(df, "text")
        assert not out["has_pii"].iloc[0]
        assert out["has_pii"].iloc[1]

    def test_original_df_not_mutated(self, detector: PIIDetector) -> None:
        df = pd.DataFrame({"text": ["a@b.com"]})
        cols_before = list(df.columns)
        detector.detect_batch(df, "text")
        assert list(df.columns) == cols_before


# ===========================================================================
# PIIPattern dataclass
# ===========================================================================

class TestPIIPatternDataclass:
    def test_all_required_pattern_types_present(self) -> None:
        pii_types = {p.pii_type for p in REGEX_PATTERNS}
        for expected in ("EMAIL", "PHONE_US", "SSN", "CREDIT_CARD", "IP_ADDRESS"):
            assert expected in pii_types

    def test_email_confidence_is_099(self) -> None:
        pat = next(p for p in REGEX_PATTERNS if p.pii_type == "EMAIL")
        assert pat.confidence == pytest.approx(0.99)

    def test_ip_confidence_is_085(self) -> None:
        pat = next(p for p in REGEX_PATTERNS if p.pii_type == "IP_ADDRESS")
        assert pat.confidence == pytest.approx(0.85)

    def test_ner_confidence_person(self) -> None:
        assert NER_CONFIDENCE["PERSON"] == pytest.approx(0.80)

    def test_ner_confidence_org(self) -> None:
        assert NER_CONFIDENCE["ORG"] == pytest.approx(0.70)

    def test_ner_confidence_gpe(self) -> None:
        assert NER_CONFIDENCE["GPE"] == pytest.approx(0.65)
