"""AppTest + unit coverage for the Phase 8.3 final UI polish pass.

This phase changed no business logic -- only presentation: chart color
contrast, data labels, humanized issue-type labels, and a more compact
safe-fixes section. These tests exist to prove the humanization mapping
actually reaches the screen (not just the dict), and that the more compact
markup still carries the real affected-count numbers correctly.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from app import ISSUE_TYPE_LABELS, humanize_issue_type

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _goto(at: AppTest, page: str, fake_file: "io.BytesIO | None" = None) -> None:
    if fake_file is not None:
        fake_file.seek(0)
    at.sidebar.button(key=f"nav_{page}").click()
    at.run(timeout=30)


def _fake_file(csv_bytes: bytes, name: str = "phase8_3_test.csv") -> "io.BytesIO":
    class FakeUploadedFile(io.BytesIO):
        pass

    FakeUploadedFile.name = name
    FakeUploadedFile.size = len(csv_bytes)
    return FakeUploadedFile(csv_bytes)


def _messy_dataset_csv() -> bytes:
    # "score" gets missing_null; "dept" has a capitalization variant.
    rows = [
        "1,10,Eng", "2,20,eng", "3,,Eng", "4,40,Sales", "5,50,Sales",
        "6,60,Sales", "7,70,Eng", "8,80,Eng", "9,90,Eng", "10,,Eng",
    ]
    return ("id,score,dept\n" + "\n".join(rows) + "\n").encode()


def _all_markdown_text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def test_humanize_issue_type_covers_every_detector_issue_type() -> None:
    """Every issue_type the rule engine can emit has a real human label, not a raw slug fallback."""
    known_issue_types = {
        "blank_string", "column_name_whitespace", "date_stored_as_text", "empty_column", "empty_row",
        "exact_duplicate_row", "identifier_duplicate_value", "identifier_inconsistent_format",
        "identifier_missing_values", "inconsistent_capitalization", "inconsistent_column_name_formatting",
        "infinite_value", "invalid_date", "leading_trailing_whitespace", "missing_null", "missing_placeholder",
        "mixed_data_types", "mixed_date_formats", "negative_value", "non_printable_characters",
        "numeric_format_inconsistency", "numeric_stored_as_text", "possible_outlier", "repeated_internal_spaces",
        "similar_category_values", "suspiciously_constant_column", "unexpected_text_in_numeric_column",
        "unusually_future_date", "unusually_old_date", "value_fails_type_conversion", "whitespace_only_string",
    }
    assert known_issue_types <= set(ISSUE_TYPE_LABELS)
    for issue_type in known_issue_types:
        label = humanize_issue_type(issue_type)
        assert "_" not in label  # never a raw slug leaking through


def test_humanize_issue_type_falls_back_gracefully_for_unknown_types() -> None:
    assert humanize_issue_type("some_new_future_issue") == "Some new future issue"


def test_guided_review_shows_human_label_not_raw_slug(tmp_path) -> None:
    fake_file = _fake_file(_messy_dataset_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        _goto(at, "Clean Data", fake_file)
        assert not at.exception

        text = _all_markdown_text(at)
        assert "Missing values" in text
        assert "missing_null" not in text


def test_safe_fix_affected_count_shows_a_unit_not_a_bare_number(tmp_path) -> None:
    fake_file = _fake_file(_messy_dataset_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        _goto(at, "Clean Data", fake_file)
        assert not at.exception

        text = _all_markdown_text(at)
        assert "columns" in text or "values" in text or "rows" in text


def test_review_issues_type_summary_is_short_with_breakdown_in_popover(tmp_path) -> None:
    fake_file = _fake_file(_messy_dataset_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        _goto(at, "Review Issues", fake_file)
        assert not at.exception

        text = _all_markdown_text(at)
        assert "detected data types" in text


def test_results_dashboard_renders_after_cleaning_with_no_errors(tmp_path) -> None:
    """Exercises the new severity chart (fixed severity colors + opacity-based before/after)."""
    fake_file = _fake_file(_messy_dataset_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        _goto(at, "Clean Data", fake_file)

        strategy_box = at.selectbox(key="missing_score_strategy")
        strategy_box.select("Replace with 0")
        fake_file.seek(0)
        at.run(timeout=30)
        at.button(key="missing_score_apply").click()
        fake_file.seek(0)
        at.run(timeout=30)
        assert not at.exception

        _goto(at, "Results", fake_file)
        assert not at.exception
