"""
Unit tests for QualityScorer.

Required test (per spec)
------------------------
test_perfect_score – a clean, complete, PII-free record must score >= 0.95
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.quality_scorer import (
    WEIGHTS,
    QualityScorer,
    _HIGH_QUALITY_THRESHOLD,
    _MIN_TEXT_LEN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scorer() -> QualityScorer:
    return QualityScorer()


def _make_row(**kwargs) -> pd.DataFrame:
    """Return a one-row DataFrame with sensible defaults for every scored column."""
    defaults: dict = {
        "text": "This is a clean record with plenty of content.",
        "timestamp": pd.Timestamp("2024-06-15", tz="UTC"),
        "source_lineage_tag": "slack::messages.json::2024-06-15 00:00:00+00:00",
        "has_pii": False,
        "pii_count": 0,
        "anonymization_count": 0,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


@pytest.fixture
def perfect_df() -> pd.DataFrame:
    """A single record that should score as close to 1.0 as possible."""
    return _make_row()


@pytest.fixture
def pii_free_multi_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "This is a well-formed record with enough characters.",
                "Another complete and consistent record here.",
                "Short",
            ],
            "timestamp": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2023-06-15", tz="UTC"),
                pd.Timestamp("2022-11-30", tz="UTC"),
            ],
            "source_lineage_tag": [
                "csv::records.csv::2024-01-01",
                "slack::msgs.json::2023-06-15",
                "plaintext::comments.txt::2022-11-30",
            ],
            "has_pii": [False, False, False],
            "pii_count": [0, 0, 0],
            "anonymization_count": [0, 0, 0],
        }
    )


# ===========================================================================
# Required test
# ===========================================================================

def test_perfect_score(scorer: QualityScorer, perfect_df: pd.DataFrame) -> None:
    """A clean, complete, PII-free record must achieve quality_score >= 0.95."""
    out = scorer.score_batch(perfect_df)
    score = out["quality_score"].iloc[0]
    assert score >= 0.95, (
        f"Expected quality_score >= 0.95 for a perfect record, got {score:.4f}. "
        f"Dimension scores: "
        f"completeness={out['completeness_score'].iloc[0]:.2f}, "
        f"consistency={out['consistency_score'].iloc[0]:.2f}, "
        f"format_conformance={out['format_conformance_score'].iloc[0]:.2f}, "
        f"pii_removal={out['pii_removal_confidence_score'].iloc[0]:.2f}"
    )


# ===========================================================================
# score_batch — output shape
# ===========================================================================

class TestScoreBatchShape:
    SCORE_COLS = [
        "completeness_score",
        "consistency_score",
        "format_conformance_score",
        "pii_removal_confidence_score",
        "quality_score",
    ]

    def test_adds_five_columns(self, scorer: QualityScorer, perfect_df: pd.DataFrame) -> None:
        out = scorer.score_batch(perfect_df)
        for col in self.SCORE_COLS:
            assert col in out.columns, f"Missing column: {col}"

    def test_original_df_not_mutated(self, scorer: QualityScorer, perfect_df: pd.DataFrame) -> None:
        cols_before = list(perfect_df.columns)
        scorer.score_batch(perfect_df)
        assert list(perfect_df.columns) == cols_before

    def test_all_scores_in_unit_interval(
        self, scorer: QualityScorer, pii_free_multi_df: pd.DataFrame
    ) -> None:
        out = scorer.score_batch(pii_free_multi_df)
        for col in self.SCORE_COLS:
            assert out[col].between(0.0, 1.0).all(), (
                f"Column {col} contains values outside [0, 1]: {out[col].tolist()}"
            )

    def test_scores_are_float_dtype(
        self, scorer: QualityScorer, perfect_df: pd.DataFrame
    ) -> None:
        out = scorer.score_batch(perfect_df)
        for col in self.SCORE_COLS:
            assert pd.api.types.is_float_dtype(out[col]), f"{col} is not float dtype"

    def test_row_count_preserved(
        self, scorer: QualityScorer, pii_free_multi_df: pd.DataFrame
    ) -> None:
        out = scorer.score_batch(pii_free_multi_df)
        assert len(out) == len(pii_free_multi_df)

    def test_composite_is_weighted_sum_of_dimensions(
        self, scorer: QualityScorer, perfect_df: pd.DataFrame
    ) -> None:
        out = scorer.score_batch(perfect_df)
        expected = (
            WEIGHTS["completeness"] * out["completeness_score"]
            + WEIGHTS["consistency"] * out["consistency_score"]
            + WEIGHTS["format_conformance"] * out["format_conformance_score"]
            + WEIGHTS["pii_removal"] * out["pii_removal_confidence_score"]
        ).clip(0.0, 1.0)
        pd.testing.assert_series_equal(
            out["quality_score"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


# ===========================================================================
# Dimension 1 — Completeness
# ===========================================================================

class TestCompletenessScore:
    def test_long_text_scores_one(self, scorer: QualityScorer) -> None:
        df = _make_row(text="This text is definitely longer than ten characters.")
        out = scorer.score_batch(df)
        assert out["completeness_score"].iloc[0] == pytest.approx(1.0)

    def test_short_text_scores_half(self, scorer: QualityScorer) -> None:
        df = _make_row(text="Hi")   # len 2, <= 10
        out = scorer.score_batch(df)
        assert out["completeness_score"].iloc[0] == pytest.approx(0.5)

    def test_exactly_ten_chars_scores_half(self, scorer: QualityScorer) -> None:
        df = _make_row(text="0123456789")   # len == 10, not > 10
        out = scorer.score_batch(df)
        assert out["completeness_score"].iloc[0] == pytest.approx(0.5)

    def test_eleven_chars_scores_one(self, scorer: QualityScorer) -> None:
        df = _make_row(text="01234567890")  # len 11 > 10
        out = scorer.score_batch(df)
        assert out["completeness_score"].iloc[0] == pytest.approx(1.0)

    def test_empty_text_scores_zero(self, scorer: QualityScorer) -> None:
        df = _make_row(text="")
        out = scorer.score_batch(df)
        assert out["completeness_score"].iloc[0] == pytest.approx(0.0)

    def test_null_text_scores_zero(self, scorer: QualityScorer) -> None:
        df = _make_row(text=None)
        out = scorer.score_batch(df)
        assert out["completeness_score"].iloc[0] == pytest.approx(0.0)

    def test_missing_text_column_scores_zero(self, scorer: QualityScorer) -> None:
        df = pd.DataFrame([{"timestamp": pd.Timestamp("2024-01-01", tz="UTC")}])
        out = scorer.score_batch(df)
        assert out["completeness_score"].iloc[0] == pytest.approx(0.0)

    def test_vectorised_across_multiple_rows(self, scorer: QualityScorer) -> None:
        df = pd.DataFrame({"text": [
            "Long enough text here",   # > 10 → 1.0
            "Short",                   # ≤ 10 → 0.5
            "",                        # empty → 0.0
        ]})
        out = scorer.score_batch(df)
        expected = [1.0, 0.5, 0.0]
        for i, exp in enumerate(expected):
            assert out["completeness_score"].iloc[i] == pytest.approx(exp)


# ===========================================================================
# Dimension 2 — Consistency
# ===========================================================================

class TestConsistencyScore:
    def test_valid_timestamp_in_range_scores_one(self, scorer: QualityScorer) -> None:
        df = _make_row(timestamp=pd.Timestamp("2024-03-10", tz="UTC"))
        out = scorer.score_batch(df)
        assert out["consistency_score"].iloc[0] == pytest.approx(1.0)

    def test_timestamp_before_range_scores_half(self, scorer: QualityScorer) -> None:
        df = _make_row(timestamp=pd.Timestamp("2005-01-01", tz="UTC"))
        out = scorer.score_batch(df)
        assert out["consistency_score"].iloc[0] == pytest.approx(0.5)

    def test_timestamp_after_range_scores_half(self, scorer: QualityScorer) -> None:
        df = _make_row(timestamp=pd.Timestamp("2035-01-01", tz="UTC"))
        out = scorer.score_batch(df)
        assert out["consistency_score"].iloc[0] == pytest.approx(0.5)

    def test_null_timestamp_scores_zero(self, scorer: QualityScorer) -> None:
        df = _make_row(timestamp=None)
        out = scorer.score_batch(df)
        assert out["consistency_score"].iloc[0] == pytest.approx(0.0)

    def test_unparseable_string_scores_zero(self, scorer: QualityScorer) -> None:
        # NaN after coerce → treated as missing → 0.0
        df = pd.DataFrame([{
            "timestamp": "not-a-date",
            "text": "irrelevant",
            "source_lineage_tag": "a::b::c",
            "has_pii": False,
            "pii_count": 0,
            "anonymization_count": 0,
        }])
        out = scorer.score_batch(df)
        assert out["consistency_score"].iloc[0] == pytest.approx(0.0)

    def test_missing_timestamp_column_scores_zero(self, scorer: QualityScorer) -> None:
        df = pd.DataFrame([{"text": "some text here that is long enough"}])
        out = scorer.score_batch(df)
        assert out["consistency_score"].iloc[0] == pytest.approx(0.0)

    def test_boundary_year_2010_scores_one(self, scorer: QualityScorer) -> None:
        df = _make_row(timestamp=pd.Timestamp("2010-01-01", tz="UTC"))
        out = scorer.score_batch(df)
        assert out["consistency_score"].iloc[0] == pytest.approx(1.0)

    def test_boundary_year_2030_scores_one(self, scorer: QualityScorer) -> None:
        df = _make_row(timestamp=pd.Timestamp("2030-12-31", tz="UTC"))
        out = scorer.score_batch(df)
        assert out["consistency_score"].iloc[0] == pytest.approx(1.0)


# ===========================================================================
# Dimension 3 — Format conformance
# ===========================================================================

class TestFormatConformanceScore:
    def test_two_separators_scores_one(self, scorer: QualityScorer) -> None:
        df = _make_row(source_lineage_tag="slack::messages.json::2024-01-01")
        out = scorer.score_batch(df)
        assert out["format_conformance_score"].iloc[0] == pytest.approx(1.0)

    def test_three_separators_still_scores_one(self, scorer: QualityScorer) -> None:
        # extra "::" in the timestamp portion is fine
        df = _make_row(source_lineage_tag="csv::file.csv::2024-01-01::extra")
        out = scorer.score_batch(df)
        assert out["format_conformance_score"].iloc[0] == pytest.approx(1.0)

    def test_one_separator_scores_half(self, scorer: QualityScorer) -> None:
        df = _make_row(source_lineage_tag="slack::messages.json")
        out = scorer.score_batch(df)
        assert out["format_conformance_score"].iloc[0] == pytest.approx(0.5)

    def test_empty_tag_scores_zero(self, scorer: QualityScorer) -> None:
        df = _make_row(source_lineage_tag="")
        out = scorer.score_batch(df)
        assert out["format_conformance_score"].iloc[0] == pytest.approx(0.0)

    def test_null_tag_scores_zero(self, scorer: QualityScorer) -> None:
        df = _make_row(source_lineage_tag=None)
        out = scorer.score_batch(df)
        assert out["format_conformance_score"].iloc[0] == pytest.approx(0.0)

    def test_no_separator_scores_zero(self, scorer: QualityScorer) -> None:
        df = _make_row(source_lineage_tag="nocolons")
        out = scorer.score_batch(df)
        assert out["format_conformance_score"].iloc[0] == pytest.approx(0.0)

    def test_missing_column_scores_zero(self, scorer: QualityScorer) -> None:
        df = pd.DataFrame([{"text": "long enough text here for sure"}])
        out = scorer.score_batch(df)
        assert out["format_conformance_score"].iloc[0] == pytest.approx(0.0)


# ===========================================================================
# Dimension 4 — PII removal confidence
# ===========================================================================

class TestPIIRemovalScore:
    def test_no_pii_scores_one(self, scorer: QualityScorer) -> None:
        df = _make_row(has_pii=False, pii_count=0, anonymization_count=0)
        out = scorer.score_batch(df)
        assert out["pii_removal_confidence_score"].iloc[0] == pytest.approx(1.0)

    def test_pii_fully_anonymized_scores_095(self, scorer: QualityScorer) -> None:
        df = _make_row(has_pii=True, pii_count=3, anonymization_count=3)
        out = scorer.score_batch(df)
        assert out["pii_removal_confidence_score"].iloc[0] == pytest.approx(0.95)

    def test_pii_over_anonymized_still_095(self, scorer: QualityScorer) -> None:
        # anonymization_count > pii_count should still count as fully anonymized
        df = _make_row(has_pii=True, pii_count=2, anonymization_count=5)
        out = scorer.score_batch(df)
        assert out["pii_removal_confidence_score"].iloc[0] == pytest.approx(0.95)

    def test_pii_partial_anonymization_scores_half(self, scorer: QualityScorer) -> None:
        df = _make_row(has_pii=True, pii_count=4, anonymization_count=2)
        out = scorer.score_batch(df)
        assert out["pii_removal_confidence_score"].iloc[0] == pytest.approx(0.5)

    def test_pii_not_anonymized_scores_zero(self, scorer: QualityScorer) -> None:
        df = _make_row(has_pii=True, pii_count=2, anonymization_count=0)
        out = scorer.score_batch(df)
        assert out["pii_removal_confidence_score"].iloc[0] == pytest.approx(0.0)

    def test_missing_pii_columns_defaults_to_one(self, scorer: QualityScorer) -> None:
        # No pii columns → treated as PII-free → score 1.0
        df = pd.DataFrame([{"text": "some long enough text here now"}])
        out = scorer.score_batch(df)
        assert out["pii_removal_confidence_score"].iloc[0] == pytest.approx(1.0)

    def test_vectorised_all_four_cases(self, scorer: QualityScorer) -> None:
        df = pd.DataFrame({
            "text": ["x" * 20] * 4,
            "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")] * 4,
            "source_lineage_tag": ["a::b::c"] * 4,
            "has_pii":            [False, True,  True, True],
            "pii_count":          [0,     3,     4,    2],
            "anonymization_count":[0,     3,     1,    0],
        })
        out = scorer.score_batch(df)
        expected = [1.0, 0.95, 0.5, 0.0]
        for i, exp in enumerate(expected):
            assert out["pii_removal_confidence_score"].iloc[i] == pytest.approx(exp), (
                f"Row {i}: expected {exp}, got {out['pii_removal_confidence_score'].iloc[i]}"
            )


# ===========================================================================
# Composite score
# ===========================================================================

class TestCompositeScore:
    def test_worst_case_scores_zero(self, scorer: QualityScorer) -> None:
        df = _make_row(
            text="",
            timestamp=None,
            source_lineage_tag="",
            has_pii=True,
            pii_count=2,
            anonymization_count=0,
        )
        out = scorer.score_batch(df)
        assert out["quality_score"].iloc[0] == pytest.approx(0.0)

    def test_weights_sum_to_one(self) -> None:
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_quality_score_monotone_with_improvement(self, scorer: QualityScorer) -> None:
        bad = scorer.score_batch(_make_row(text="Hi", timestamp=None, source_lineage_tag=""))
        good = scorer.score_batch(_make_row())
        assert good["quality_score"].iloc[0] > bad["quality_score"].iloc[0]


# ===========================================================================
# generate_report
# ===========================================================================

class TestGenerateReport:
    @pytest.fixture
    def scored_df(self, scorer: QualityScorer, pii_free_multi_df: pd.DataFrame) -> pd.DataFrame:
        return scorer.score_batch(pii_free_multi_df)

    def test_returns_dict(
        self, scorer: QualityScorer, scored_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        report = scorer.generate_report(scored_df, str(tmp_path / "report.json"))
        assert isinstance(report, dict)

    def test_required_keys_present(
        self, scorer: QualityScorer, scored_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        report = scorer.generate_report(scored_df, str(tmp_path / "report.json"))
        for key in (
            "record_count",
            "mean_quality_score",
            "median_quality_score",
            "std_quality_score",
            "pct_above_threshold",
            "pct_records_with_pii",
            "pct_pii_records_fully_anonymized",
            "weights",
        ):
            assert key in report, f"Missing key in report: {key}"

    def test_json_file_written(
        self, scorer: QualityScorer, scored_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out_path = tmp_path / "sub" / "report.json"
        scorer.generate_report(scored_df, str(out_path))
        assert out_path.exists()
        # File must be valid JSON
        payload = json.loads(out_path.read_text())
        assert "mean_quality_score" in payload

    def test_parent_dirs_created(
        self, scorer: QualityScorer, scored_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out_path = tmp_path / "a" / "b" / "c" / "report.json"
        scorer.generate_report(scored_df, str(out_path))
        assert out_path.exists()

    def test_record_count_correct(
        self, scorer: QualityScorer, scored_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        report = scorer.generate_report(scored_df, str(tmp_path / "r.json"))
        assert report["record_count"] == len(scored_df)

    def test_mean_score_in_unit_interval(
        self, scorer: QualityScorer, scored_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        report = scorer.generate_report(scored_df, str(tmp_path / "r.json"))
        assert 0.0 <= report["mean_quality_score"] <= 1.0

    def test_pct_above_threshold_for_all_perfect(
        self, scorer: QualityScorer, tmp_path: Path
    ) -> None:
        df = scorer.score_batch(pd.DataFrame([_make_row().iloc[0]] * 5))
        report = scorer.generate_report(df, str(tmp_path / "r.json"))
        assert report["pct_above_threshold"] == pytest.approx(100.0)

    def test_missing_quality_score_col_raises(
        self, scorer: QualityScorer, tmp_path: Path
    ) -> None:
        df = pd.DataFrame([{"text": "hello"}])
        with pytest.raises(ValueError, match="quality_score"):
            scorer.generate_report(df, str(tmp_path / "r.json"))

    def test_std_is_zero_for_single_row(
        self, scorer: QualityScorer, tmp_path: Path
    ) -> None:
        df = scorer.score_batch(_make_row())
        report = scorer.generate_report(df, str(tmp_path / "r.json"))
        assert report["std_quality_score"] == pytest.approx(0.0)

    def test_weights_embedded_in_report(
        self, scorer: QualityScorer, scored_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        report = scorer.generate_report(scored_df, str(tmp_path / "r.json"))
        assert report["weights"] == WEIGHTS
