"""
Unit tests for the Anonymizer engine.

Required tests (per spec)
--------------------------
test_email_pseudonymized  – [EMAIL_XXXXXXXX] token produced for email input
test_clean_text_unchanged – text with no PII detections is returned unmodified
"""

import hashlib
import re

import pandas as pd
import pytest

from pipeline.anonymizer import (
    PSEUDONYMIZE_TYPES,
    REDACT_TYPES,
    Anonymizer,
    make_pseudo_token,
    make_redact_token,
)
from pipeline.pii_detector import PIIDetector


# ---------------------------------------------------------------------------
# Shared fixtures  (spaCy loaded once per session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector() -> PIIDetector:
    return PIIDetector()


@pytest.fixture(scope="module")
def anonymizer() -> Anonymizer:
    return Anonymizer()


# ===========================================================================
# Required tests
# ===========================================================================

def test_email_pseudonymized(detector: PIIDetector, anonymizer: Anonymizer) -> None:
    """EMAIL must produce a [EMAIL_XXXXXXXX] pseudo-token in the output."""
    text = "Contact me at john@example.com for details."
    detections = detector.detect(text)
    anon_text, _ = anonymizer.anonymize(text, detections)
    assert "[EMAIL_" in anon_text, (
        f"Expected [EMAIL_...] token in anonymized output, got: {anon_text!r}"
    )


def test_clean_text_unchanged(detector: PIIDetector, anonymizer: Anonymizer) -> None:
    """Text that contains no PII must pass through without modification."""
    text = "The pipeline ran successfully at 3pm."
    detections = detector.detect(text)
    anon_text, log = anonymizer.anonymize(text, detections)
    assert anon_text == text, (
        f"Clean text should be unchanged, got: {anon_text!r}"
    )
    assert log == [], f"Expected empty log for clean text, got: {log}"


# ===========================================================================
# Token format
# ===========================================================================

class TestPseudoTokenFormat:
    """make_pseudo_token must produce exactly [TYPE_XXXXXXXX] with 8 hex chars."""

    def test_format_structure(self) -> None:
        token = make_pseudo_token("EMAIL", "john@example.com")
        assert re.fullmatch(r"\[EMAIL_[0-9a-f]{8}\]", token), (
            f"Token does not match [EMAIL_XXXXXXXX] pattern: {token!r}"
        )

    def test_hash_is_first_8_chars_of_sha256(self) -> None:
        matched = "john@example.com"
        expected_digest = hashlib.sha256(matched.encode("utf-8")).hexdigest()[:8]
        token = make_pseudo_token("EMAIL", matched)
        assert token == f"[EMAIL_{expected_digest}]"

    def test_deterministic_same_input_same_token(self) -> None:
        t1 = make_pseudo_token("SSN", "123-45-6789")
        t2 = make_pseudo_token("SSN", "123-45-6789")
        assert t1 == t2

    def test_different_inputs_different_tokens(self) -> None:
        t1 = make_pseudo_token("EMAIL", "alice@example.com")
        t2 = make_pseudo_token("EMAIL", "bob@example.com")
        assert t1 != t2

    def test_pii_type_prefix_preserved(self) -> None:
        for pii_type in ("EMAIL", "PHONE_US", "SSN", "CREDIT_CARD", "PERSON"):
            token = make_pseudo_token(pii_type, "dummy")
            assert token.startswith(f"[{pii_type}_"), (
                f"Token {token!r} does not start with [{pii_type}_"
            )


class TestRedactTokenFormat:
    """make_redact_token must produce exactly [REDACTED_TYPE]."""

    def test_format_structure(self) -> None:
        assert make_redact_token("ORG") == "[REDACTED_ORG]"
        assert make_redact_token("GPE") == "[REDACTED_GPE]"
        assert make_redact_token("IP_ADDRESS") == "[REDACTED_IP_ADDRESS]"

    def test_pii_type_embedded_in_token(self) -> None:
        for pii_type in REDACT_TYPES:
            token = make_redact_token(pii_type)
            assert pii_type in token


# ===========================================================================
# Strategy routing
# ===========================================================================

class TestStrategyRouting:
    """Each PII type must be routed to the declared strategy."""

    def _detect_one(self, detector: PIIDetector, text: str, pii_type: str) -> dict:
        return next(d for d in detector.detect(text) if d["pii_type"] == pii_type)

    def test_email_routes_to_pseudonymization(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        det = self._detect_one(detector, "email: a@b.com", "EMAIL")
        _, log = anonymizer.anonymize("email: a@b.com", [det])
        assert log[0]["strategy"] == "pseudonymization"

    def test_ssn_routes_to_pseudonymization(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        det = self._detect_one(detector, "SSN 123-45-6789", "SSN")
        _, log = anonymizer.anonymize("SSN 123-45-6789", [det])
        assert log[0]["strategy"] == "pseudonymization"

    def test_ip_address_routes_to_redaction(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        det = self._detect_one(detector, "server at 10.0.0.1", "IP_ADDRESS")
        _, log = anonymizer.anonymize("server at 10.0.0.1", [det])
        assert log[0]["strategy"] == "redaction"

    def test_org_routes_to_redaction(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        dets = detector.detect("Reviewed by Microsoft Legal.")
        org_dets = [d for d in dets if d["pii_type"] == "ORG"]
        if org_dets:  # spaCy must detect ORG for this assertion to be meaningful
            _, log = anonymizer.anonymize("Reviewed by Microsoft Legal.", org_dets)
            assert all(e["strategy"] == "redaction" for e in log)

    def test_all_pseudonymize_types_declared(self) -> None:
        assert PSEUDONYMIZE_TYPES == {"EMAIL", "PHONE_US", "SSN", "CREDIT_CARD", "PERSON"}

    def test_all_redact_types_declared(self) -> None:
        assert REDACT_TYPES == {"IP_ADDRESS", "ORG", "GPE"}

    def test_type_sets_are_disjoint(self) -> None:
        assert PSEUDONYMIZE_TYPES.isdisjoint(REDACT_TYPES), (
            "A PII type cannot belong to both strategy sets"
        )


# ===========================================================================
# anonymize() — return value shape
# ===========================================================================

class TestAnonymizeReturn:
    def test_returns_tuple(self, detector: PIIDetector, anonymizer: Anonymizer) -> None:
        result = anonymizer.anonymize("email a@b.com", detector.detect("email a@b.com"))
        assert isinstance(result, tuple) and len(result) == 2

    def test_first_element_is_str(self, detector: PIIDetector, anonymizer: Anonymizer) -> None:
        anon_text, _ = anonymizer.anonymize("a@b.com", detector.detect("a@b.com"))
        assert isinstance(anon_text, str)

    def test_second_element_is_list(self, detector: PIIDetector, anonymizer: Anonymizer) -> None:
        _, log = anonymizer.anonymize("a@b.com", detector.detect("a@b.com"))
        assert isinstance(log, list)

    def test_empty_detections_returns_original(self, anonymizer: Anonymizer) -> None:
        text = "Nothing here."
        anon_text, log = anonymizer.anonymize(text, [])
        assert anon_text == text
        assert log == []

    def test_original_text_absent_after_anonymization(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        text = "Send to alice@secret.com"
        dets = detector.detect(text)
        anon_text, _ = anonymizer.anonymize(text, dets)
        assert "alice@secret.com" not in anon_text


# ===========================================================================
# anonymize() — log entry shape
# ===========================================================================

class TestLogEntryShape:
    REQUIRED_KEYS = {"original_text", "replacement", "pii_type", "strategy", "position"}

    def test_log_entry_has_all_required_keys(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        _, log = anonymizer.anonymize("a@b.com", detector.detect("a@b.com"))
        assert log, "Expected at least one log entry"
        assert self.REQUIRED_KEYS.issubset(log[0].keys())

    def test_position_has_start_and_end(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        _, log = anonymizer.anonymize("a@b.com", detector.detect("a@b.com"))
        pos = log[0]["position"]
        assert "start" in pos and "end" in pos
        assert isinstance(pos["start"], int) and isinstance(pos["end"], int)

    def test_log_original_text_matches_detection(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        text = "Contact alice@example.com"
        dets = detector.detect(text)
        email_det = next(d for d in dets if d["pii_type"] == "EMAIL")
        _, log = anonymizer.anonymize(text, [email_det])
        assert log[0]["original_text"] == email_det["matched_text"]

    def test_log_sorted_by_start_position(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        text = "SSN 123-45-6789 and email b@c.com"
        _, log = anonymizer.anonymize(text, detector.detect(text))
        starts = [e["position"]["start"] for e in log]
        assert starts == sorted(starts)

    def test_strategy_field_is_valid_string(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        _, log = anonymizer.anonymize("a@b.com", detector.detect("a@b.com"))
        assert log[0]["strategy"] in {"pseudonymization", "redaction"}


# ===========================================================================
# anonymize() — character-offset correctness
# ===========================================================================

class TestOffsetCorrectness:
    def test_single_pii_replaced_at_correct_position(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        text = "Sender: alice@example.com."
        dets = detector.detect(text)
        anon_text, log = anonymizer.anonymize(text, dets)
        # The email must be gone and its token must appear in the output
        assert "alice@example.com" not in anon_text
        assert "[EMAIL_" in anon_text
        # Non-PII literal after the email must survive
        assert anon_text.endswith(".")

    def test_multiple_pii_all_replaced(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        text = "Call 415-555-0192 or email bob@test.com."
        dets = detector.detect(text)
        anon_text, log = anonymizer.anonymize(text, dets)
        assert "415-555-0192" not in anon_text
        assert "bob@test.com" not in anon_text
        assert len(log) >= 2

    def test_text_with_pii_at_start(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        text = "alice@example.com is my address."
        dets = detector.detect(text)
        anon_text, _ = anonymizer.anonymize(text, dets)
        assert "alice@example.com" not in anon_text
        assert "is my address." in anon_text

    def test_text_with_pii_at_end(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        text = "Send the report to alice@example.com"
        dets = detector.detect(text)
        anon_text, _ = anonymizer.anonymize(text, dets)
        assert "alice@example.com" not in anon_text

    def test_determinism_across_calls(
        self, detector: PIIDetector, anonymizer: Anonymizer
    ) -> None:
        text = "Reach me at dev@corp.io for access."
        dets = detector.detect(text)
        out1, _ = anonymizer.anonymize(text, dets)
        out2, _ = anonymizer.anonymize(text, dets)
        assert out1 == out2


# ===========================================================================
# anonymize_batch()
# ===========================================================================

class TestAnonymizeBatch:
    @pytest.fixture(scope="class")
    def batch_df(self, detector: PIIDetector) -> pd.DataFrame:
        """Small DataFrame that has gone through detect_batch."""
        df = pd.DataFrame(
            {
                "text": [
                    "Email alice@example.com for access.",
                    "The build is green.",
                    "Server IP is 192.168.1.1. Contact Bob Smith.",
                ]
            }
        )
        return detector.detect_batch(df, "text")

    def test_adds_four_columns(
        self, anonymizer: Anonymizer, batch_df: pd.DataFrame
    ) -> None:
        out = anonymizer.anonymize_batch(batch_df)
        for col in ("anonymized_text", "anonymization_log", "anonymization_count", "strategies_used"):
            assert col in out.columns, f"Missing column: {col}"

    def test_original_df_not_mutated(
        self, anonymizer: Anonymizer, batch_df: pd.DataFrame
    ) -> None:
        cols_before = list(batch_df.columns)
        anonymizer.anonymize_batch(batch_df)
        assert list(batch_df.columns) == cols_before

    def test_clean_row_count_is_zero(
        self, anonymizer: Anonymizer, batch_df: pd.DataFrame
    ) -> None:
        out = anonymizer.anonymize_batch(batch_df)
        assert out["anonymization_count"].iloc[1] == 0

    def test_email_row_count_is_positive(
        self, anonymizer: Anonymizer, batch_df: pd.DataFrame
    ) -> None:
        out = anonymizer.anonymize_batch(batch_df)
        assert out["anonymization_count"].iloc[0] > 0

    def test_strategies_used_is_set(
        self, anonymizer: Anonymizer, batch_df: pd.DataFrame
    ) -> None:
        out = anonymizer.anonymize_batch(batch_df)
        for s in out["strategies_used"]:
            assert isinstance(s, set)

    def test_strategies_subset_of_valid_values(
        self, anonymizer: Anonymizer, batch_df: pd.DataFrame
    ) -> None:
        valid = {"pseudonymization", "redaction"}
        out = anonymizer.anonymize_batch(batch_df)
        for strategies in out["strategies_used"]:
            assert strategies.issubset(valid)

    def test_missing_pii_detections_raises(
        self, anonymizer: Anonymizer
    ) -> None:
        df = pd.DataFrame({"text": ["some text"]})
        with pytest.raises(ValueError, match="pii_detections"):
            anonymizer.anonymize_batch(df)

    def test_missing_text_col_raises(
        self, anonymizer: Anonymizer, batch_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="text column"):
            anonymizer.anonymize_batch(batch_df, text_col="nonexistent_col")

    def test_anonymized_text_differs_from_original_for_pii_row(
        self, anonymizer: Anonymizer, batch_df: pd.DataFrame
    ) -> None:
        out = anonymizer.anonymize_batch(batch_df)
        # Row 0 has PII — anonymized text must differ from original
        assert out["anonymized_text"].iloc[0] != out["text"].iloc[0]

    def test_anonymized_text_equals_original_for_clean_row(
        self, anonymizer: Anonymizer, batch_df: pd.DataFrame
    ) -> None:
        out = anonymizer.anonymize_batch(batch_df)
        # Row 1 is clean — anonymized text must equal original
        assert out["anonymized_text"].iloc[1] == out["text"].iloc[1]
