"""Unit tests for LineageLogger."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline.lineage_logger import LineageLogger, _make_run_id, _to_json_safe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXED_RUN_ID = "test_run_20240615"

_NOW = datetime(2024, 6, 15, 14, 30, 22, tzinfo=timezone.utc)
_NOW_ISO = _NOW.isoformat()


@pytest.fixture
def logger(tmp_path: Path) -> LineageLogger:
    return LineageLogger(str(tmp_path / "lineage"), run_id=FIXED_RUN_ID)


# ===========================================================================
# Constructor
# ===========================================================================

class TestConstructor:
    def test_output_dir_created(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "lineage"
        assert not target.exists()
        LineageLogger(str(target), run_id="r1")
        assert target.is_dir()

    def test_explicit_run_id_preserved(self, tmp_path: Path) -> None:
        lg = LineageLogger(str(tmp_path), run_id="my-run-42")
        assert lg.run_id == "my-run-42"

    def test_auto_run_id_generated_when_none(self, tmp_path: Path) -> None:
        lg = LineageLogger(str(tmp_path))
        assert lg.run_id  # non-empty
        assert isinstance(lg.run_id, str)

    def test_auto_run_ids_are_unique(self, tmp_path: Path) -> None:
        ids = {LineageLogger(str(tmp_path)).run_id for _ in range(10)}
        assert len(ids) == 10

    def test_buffer_starts_empty(self, logger: LineageLogger) -> None:
        assert len(logger) == 0
        assert logger.event_count == 0


# ===========================================================================
# _make_run_id helper
# ===========================================================================

class TestMakeRunId:
    def test_format_timestamp_prefix(self) -> None:
        run_id = _make_run_id()
        # Should match YYYYMMDD_HHMMSS_<8hex> — 8+1+6+1+8 = 24 chars
        assert len(run_id) == 24, f"Unexpected run_id length: {run_id!r}"
        date_part, time_part, uid_part = run_id.split("_")
        assert date_part.isdigit() and len(date_part) == 8
        assert time_part.isdigit() and len(time_part) == 6
        assert len(uid_part) == 8

    def test_uniqueness(self) -> None:
        ids = {_make_run_id() for _ in range(50)}
        assert len(ids) == 50


# ===========================================================================
# _to_json_safe helper
# ===========================================================================

class TestToJsonSafe:
    def test_datetime_to_iso(self) -> None:
        result = _to_json_safe(_NOW)
        assert result == _NOW_ISO

    def test_set_to_sorted_list(self) -> None:
        result = _to_json_safe({"EMAIL", "PERSON", "ORG"})
        assert result == ["EMAIL", "ORG", "PERSON"]

    def test_frozenset_to_sorted_list(self) -> None:
        result = _to_json_safe(frozenset({"B", "A"}))
        assert result == ["A", "B"]

    def test_path_to_str(self) -> None:
        p = Path("/some/path/file.txt")
        assert _to_json_safe(p) == str(p)

    def test_primitives_pass_through(self) -> None:
        for val in (42, 3.14, "hello", True, None, [1, 2]):
            assert _to_json_safe(val) == val


# ===========================================================================
# log_ingestion
# ===========================================================================

class TestLogIngestion:
    def test_appends_one_event(self, logger: LineageLogger) -> None:
        logger.log_ingestion("r1", "slack", "msgs.json", _NOW, 150)
        assert len(logger) == 1

    def test_event_type(self, logger: LineageLogger) -> None:
        logger.log_ingestion("r1", "slack", "msgs.json", _NOW, 150)
        assert logger._events[0]["event"] == "ingested"

    def test_all_required_keys_present(self, logger: LineageLogger) -> None:
        logger.log_ingestion("r1", "slack", "msgs.json", _NOW, 150)
        ev = logger._events[0]
        for key in (
            "event", "run_id", "record_id", "source_format", "source_file",
            "ingestion_timestamp", "source_timestamp", "text_length",
        ):
            assert key in ev, f"Missing key: {key}"

    def test_run_id_embedded(self, logger: LineageLogger) -> None:
        logger.log_ingestion("r1", "slack", "msgs.json", _NOW, 150)
        assert logger._events[0]["run_id"] == FIXED_RUN_ID

    def test_source_timestamp_serialised(self, logger: LineageLogger) -> None:
        logger.log_ingestion("r1", "slack", "msgs.json", _NOW, 150)
        assert logger._events[0]["source_timestamp"] == _NOW_ISO

    def test_text_length_stored(self, logger: LineageLogger) -> None:
        logger.log_ingestion("r1", "slack", "msgs.json", _NOW, 99)
        assert logger._events[0]["text_length"] == 99

    def test_ingestion_timestamp_is_iso_string(self, logger: LineageLogger) -> None:
        logger.log_ingestion("r1", "slack", "msgs.json", _NOW, 10)
        ts = logger._events[0]["ingestion_timestamp"]
        # Must be parseable back to a datetime
        datetime.fromisoformat(ts)


# ===========================================================================
# log_pii_detection
# ===========================================================================

class TestLogPiiDetection:
    def test_appends_one_event(self, logger: LineageLogger) -> None:
        logger.log_pii_detection("r1", {"EMAIL"}, 1, _NOW)
        assert len(logger) == 1

    def test_event_type(self, logger: LineageLogger) -> None:
        logger.log_pii_detection("r1", {"EMAIL"}, 1, _NOW)
        assert logger._events[0]["event"] == "pii_detected"

    def test_all_required_keys(self, logger: LineageLogger) -> None:
        logger.log_pii_detection("r1", {"EMAIL"}, 2, _NOW)
        for key in ("event", "run_id", "record_id", "pii_types_found",
                    "pii_count", "detection_timestamp"):
            assert key in logger._events[0]

    def test_set_types_sorted_to_list(self, logger: LineageLogger) -> None:
        logger.log_pii_detection("r1", {"PERSON", "EMAIL", "ORG"}, 3, _NOW)
        result = logger._events[0]["pii_types_found"]
        assert isinstance(result, list)
        assert result == sorted(result)

    def test_list_types_accepted(self, logger: LineageLogger) -> None:
        logger.log_pii_detection("r1", ["EMAIL", "SSN"], 2, _NOW)
        assert logger._events[0]["pii_types_found"] == ["EMAIL", "SSN"]

    def test_pii_count_stored(self, logger: LineageLogger) -> None:
        logger.log_pii_detection("r1", set(), 7, _NOW)
        assert logger._events[0]["pii_count"] == 7

    def test_detection_timestamp_serialised(self, logger: LineageLogger) -> None:
        logger.log_pii_detection("r1", set(), 0, _NOW)
        assert logger._events[0]["detection_timestamp"] == _NOW_ISO


# ===========================================================================
# log_anonymization
# ===========================================================================

class TestLogAnonymization:
    def test_appends_one_event(self, logger: LineageLogger) -> None:
        logger.log_anonymization("r1", "pseudonymization", "EMAIL", 20, "[EMAIL_abc12345]", _NOW)
        assert len(logger) == 1

    def test_event_type(self, logger: LineageLogger) -> None:
        logger.log_anonymization("r1", "redaction", "ORG", 9, "[REDACTED_ORG]", _NOW)
        assert logger._events[0]["event"] == "anonymized"

    def test_all_required_keys(self, logger: LineageLogger) -> None:
        logger.log_anonymization("r1", "redaction", "GPE", 8, "[REDACTED_GPE]", _NOW)
        for key in ("event", "run_id", "record_id", "strategy", "pii_type",
                    "original_length", "replacement", "anonymization_timestamp"):
            assert key in logger._events[0]

    def test_strategy_stored(self, logger: LineageLogger) -> None:
        logger.log_anonymization("r1", "pseudonymization", "SSN", 11, "[SSN_deadbeef]", _NOW)
        assert logger._events[0]["strategy"] == "pseudonymization"

    def test_replacement_token_stored(self, logger: LineageLogger) -> None:
        token = "[EMAIL_a94f2c1b]"
        logger.log_anonymization("r1", "pseudonymization", "EMAIL", 16, token, _NOW)
        assert logger._events[0]["replacement"] == token

    def test_original_length_stored(self, logger: LineageLogger) -> None:
        logger.log_anonymization("r1", "redaction", "IP_ADDRESS", 13, "[REDACTED_IP_ADDRESS]", _NOW)
        assert logger._events[0]["original_length"] == 13

    def test_multiple_calls_multiple_events(self, logger: LineageLogger) -> None:
        logger.log_anonymization("r1", "pseudonymization", "EMAIL", 16, "[EMAIL_x]", _NOW)
        logger.log_anonymization("r1", "redaction", "ORG", 9, "[REDACTED_ORG]", _NOW)
        assert len(logger) == 2


# ===========================================================================
# log_quality_score
# ===========================================================================

class TestLogQualityScore:
    def test_appends_one_event(self, logger: LineageLogger) -> None:
        logger.log_quality_score("r1", 0.92, 1.0, 0.95, True)
        assert len(logger) == 1

    def test_event_type(self, logger: LineageLogger) -> None:
        logger.log_quality_score("r1", 0.92, 1.0, 0.95, True)
        assert logger._events[0]["event"] == "quality_scored"

    def test_all_required_keys(self, logger: LineageLogger) -> None:
        logger.log_quality_score("r1", 0.5, 0.5, 0.5, False)
        for key in ("event", "run_id", "record_id", "quality_score",
                    "completeness_score", "pii_removal_confidence_score",
                    "passed_threshold"):
            assert key in logger._events[0]

    def test_scores_stored_correctly(self, logger: LineageLogger) -> None:
        logger.log_quality_score("r1", 0.88, 1.0, 0.95, True)
        ev = logger._events[0]
        assert ev["quality_score"] == pytest.approx(0.88)
        assert ev["completeness_score"] == pytest.approx(1.0)
        assert ev["pii_removal_confidence_score"] == pytest.approx(0.95)

    def test_passed_threshold_true(self, logger: LineageLogger) -> None:
        logger.log_quality_score("r1", 0.99, 1.0, 1.0, True)
        assert logger._events[0]["passed_threshold"] is True

    def test_passed_threshold_false(self, logger: LineageLogger) -> None:
        logger.log_quality_score("r1", 0.3, 0.2, 0.0, False)
        assert logger._events[0]["passed_threshold"] is False

    def test_run_id_matches(self, logger: LineageLogger) -> None:
        logger.log_quality_score("r1", 0.75, 0.8, 0.7, False)
        assert logger._events[0]["run_id"] == FIXED_RUN_ID


# ===========================================================================
# flush — JSONL output
# ===========================================================================

class TestFlushJSONL:
    def _populate(self, logger: LineageLogger) -> None:
        logger.log_ingestion("r1", "slack", "f.json", _NOW, 80)
        logger.log_pii_detection("r1", {"EMAIL"}, 1, _NOW)
        logger.log_anonymization("r1", "pseudonymization", "EMAIL", 16, "[EMAIL_x]", _NOW)
        logger.log_quality_score("r1", 0.95, 1.0, 0.95, True)

    def test_returns_string_path(self, logger: LineageLogger) -> None:
        self._populate(logger)
        result = logger.flush()
        assert isinstance(result, str)

    def test_jsonl_file_created(self, logger: LineageLogger) -> None:
        self._populate(logger)
        path = logger.flush()
        assert Path(path).exists()

    def test_correct_line_count(self, logger: LineageLogger) -> None:
        self._populate(logger)
        path = logger.flush()
        lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4

    def test_every_line_is_valid_json(self, logger: LineageLogger) -> None:
        self._populate(logger)
        path = logger.flush()
        for line in Path(path).read_text().strip().splitlines():
            obj = json.loads(line)
            assert isinstance(obj, dict)

    def test_event_order_preserved(self, logger: LineageLogger) -> None:
        self._populate(logger)
        path = logger.flush()
        events = [json.loads(l)["event"] for l in Path(path).read_text().strip().splitlines()]
        assert events == ["ingested", "pii_detected", "anonymized", "quality_scored"]

    def test_buffer_cleared_after_flush(self, logger: LineageLogger) -> None:
        self._populate(logger)
        logger.flush()
        assert len(logger) == 0

    def test_custom_filename_honoured(self, logger: LineageLogger) -> None:
        self._populate(logger)
        path = logger.flush(filename="my_run.jsonl")
        assert Path(path).name == "my_run.jsonl"

    def test_default_filename(self, logger: LineageLogger) -> None:
        self._populate(logger)
        path = logger.flush()
        assert Path(path).name == "lineage_log.jsonl"

    def test_run_id_in_every_event(self, logger: LineageLogger) -> None:
        self._populate(logger)
        path = logger.flush()
        for line in Path(path).read_text().strip().splitlines():
            assert json.loads(line)["run_id"] == FIXED_RUN_ID

    def test_empty_flush_produces_empty_jsonl(self, logger: LineageLogger) -> None:
        path = logger.flush()
        content = Path(path).read_text().strip()
        assert content == ""


# ===========================================================================
# flush — summary sidecar
# ===========================================================================

class TestFlushSummary:
    def _flush_with_two_records(self, logger: LineageLogger) -> dict:
        logger.log_ingestion("r1", "slack", "f.json", _NOW, 80)
        logger.log_ingestion("r2", "csv", "g.csv", _NOW, 50)
        logger.log_pii_detection("r1", {"EMAIL"}, 1, _NOW)
        logger.log_quality_score("r1", 0.9, 1.0, 0.9, True)
        logger.log_quality_score("r2", 0.6, 0.5, 0.5, False)
        path = logger.flush()
        summary_path = Path(path).parent / "lineage_log_summary.json"
        return json.loads(summary_path.read_text())

    def test_summary_file_created(self, logger: LineageLogger) -> None:
        logger.log_ingestion("r1", "slack", "f.json", _NOW, 10)
        path = logger.flush()
        summary_path = Path(path).parent / "lineage_log_summary.json"
        assert summary_path.exists()

    def test_summary_has_required_keys(self, logger: LineageLogger) -> None:
        summary = self._flush_with_two_records(logger)
        for key in ("run_id", "total_events", "events_by_type",
                    "records_processed", "log_file"):
            assert key in summary, f"Missing summary key: {key}"

    def test_total_events_correct(self, logger: LineageLogger) -> None:
        summary = self._flush_with_two_records(logger)
        assert summary["total_events"] == 5

    def test_events_by_type_correct(self, logger: LineageLogger) -> None:
        summary = self._flush_with_two_records(logger)
        assert summary["events_by_type"]["ingested"] == 2
        assert summary["events_by_type"]["pii_detected"] == 1
        assert summary["events_by_type"]["quality_scored"] == 2

    def test_records_processed_counts_unique_ids(self, logger: LineageLogger) -> None:
        summary = self._flush_with_two_records(logger)
        assert summary["records_processed"] == 2

    def test_run_id_in_summary(self, logger: LineageLogger) -> None:
        summary = self._flush_with_two_records(logger)
        assert summary["run_id"] == FIXED_RUN_ID

    def test_log_file_path_in_summary(self, logger: LineageLogger) -> None:
        summary = self._flush_with_two_records(logger)
        assert Path(summary["log_file"]).exists()

    def test_custom_filename_reflected_in_summary_name(
        self, logger: LineageLogger
    ) -> None:
        logger.log_ingestion("r1", "slack", "f.json", _NOW, 10)
        path = logger.flush(filename="audit_run.jsonl")
        summary_path = Path(path).parent / "audit_run_summary.json"
        assert summary_path.exists()

    def test_summary_is_valid_json(self, logger: LineageLogger) -> None:
        logger.log_ingestion("r1", "slack", "f.json", _NOW, 10)
        path = logger.flush()
        summary_path = Path(path).parent / "lineage_log_summary.json"
        obj = json.loads(summary_path.read_text())
        assert isinstance(obj, dict)


# ===========================================================================
# repr and properties
# ===========================================================================

class TestReprAndProperties:
    def test_repr_contains_run_id(self, logger: LineageLogger) -> None:
        assert FIXED_RUN_ID in repr(logger)

    def test_len_matches_event_count(self, logger: LineageLogger) -> None:
        logger.log_ingestion("r1", "slack", "f.json", _NOW, 10)
        logger.log_quality_score("r1", 0.9, 1.0, 0.9, True)
        assert len(logger) == logger.event_count == 2
