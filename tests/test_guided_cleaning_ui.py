"""End-to-end AppTest smoke test for the Phase 7B guided cleaning workflow.

Complements the pure-function tests in test_cleaning_strategies.py,
test_cleaning_engine.py, and test_issue_grouping.py (which cover the
business logic in isolation) by driving the actual Streamlit widgets:
selecting a strategy, confirming a preview, and clicking "Apply Cleaning
Decision" -- verifying the UI wiring itself (session state updates,
audit log, and the separate cleaning decision log) without a browser.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _fake_uploaded_csv() -> "io.BytesIO":
    # 10 rows, 2 missing "score" values -- exactly one guided-review group
    # (score / missing_null) should appear; "dept" has no case/spacing
    # variants and "id" is fully populated, so nothing else is flagged.
    rows = [
        "1,10,Eng", "2,20,Eng", "3,,Eng", "4,40,Sales", "5,50,Sales",
        "6,60,Sales", "7,70,Eng", "8,80,Eng", "9,90,Eng", "10,,Eng",
    ]
    csv_bytes = ("id,score,dept\n" + "\n".join(rows) + "\n").encode()

    class FakeUploadedFile(io.BytesIO):
        name = "guided_smoke_test.csv"
        size = len(csv_bytes)

    return FakeUploadedFile(csv_bytes)


def test_keep_as_null_decision_is_logged_without_changing_data(tmp_path) -> None:
    fake_file = _fake_uploaded_csv()
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        at.sidebar.button(key="nav_Clean Data").click()
        fake_file.seek(0)
        at.run(timeout=30)
        assert not at.exception

        strategy_box = at.selectbox(key="missing_score_strategy")
        strategy_box.select("Keep as NULL")
        fake_file.seek(0)
        at.run(timeout=30)
        assert not at.exception

        apply_button = at.button(key="missing_score_apply")
        apply_button.click()
        fake_file.seek(0)
        at.run(timeout=30)
        assert not at.exception

        decision_log = at.session_state["cleaning_decision_log"]
        assert len(decision_log) == 1
        entry = decision_log[0]
        assert entry.column_name == "score"
        assert entry.issue_type == "missing_null"
        assert entry.selected_strategy == "Keep as NULL"
        assert entry.decision_result == "Keep as NULL"
        assert entry.affected_count == 2

        cleaning_result = at.session_state["cleaning_result"]
        assert cleaning_result.audit_log == []  # Keep as NULL changes no data
        assert cleaning_result.cleaned_df["score"].isna().sum() == 2


def test_replace_with_zero_decision_changes_only_the_working_copy(tmp_path) -> None:
    fake_file = _fake_uploaded_csv()
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        at.sidebar.button(key="nav_Clean Data").click()
        fake_file.seek(0)
        at.run(timeout=30)

        strategy_box = at.selectbox(key="missing_score_strategy")
        strategy_box.select("Replace with 0")
        fake_file.seek(0)
        at.run(timeout=30)
        assert not at.exception

        apply_button = at.button(key="missing_score_apply")
        apply_button.click()
        fake_file.seek(0)
        at.run(timeout=30)
        assert not at.exception

        cleaning_result = at.session_state["cleaning_result"]
        assert len(cleaning_result.audit_log) == 2
        assert cleaning_result.cleaned_df["score"].isna().sum() == 0
        assert (cleaning_result.cleaned_df.loc[cleaning_result.cleaned_df["id"].isin([3, 10]), "score"] == 0).all()

        decision_log = at.session_state["cleaning_decision_log"]
        assert len(decision_log) == 1
        assert decision_log[0].selected_strategy == "Replace with 0"
        assert decision_log[0].affected_count == 2
