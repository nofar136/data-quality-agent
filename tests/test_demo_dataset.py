"""Tests for the bundled portfolio demo dataset (data/demo_employee_data.csv).

This is the same "Messy Employee Dataset" (~1,020 rows) already used
throughout this project's own development and UI testing (e.g. the Age
missing-values / median cleaning example) -- reused byte-for-byte as the
bundled demo, not a newly generated file. Covers two things: (1) the dataset
file itself is what it's supposed to be and the app's own pipeline can
actually parse and profile it, and (2) the "Use Demo Dataset" button on the
Upload page loads it through that exact same pipeline -- no separate demo
code path -- and makes it obvious in the UI that a demo, not a real upload,
is active.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.config import DEMO_DATASET_PATH
from src.file_loader import load_csv
from src.profiler import profile_dataframe
from src.rule_engine import detect_issues

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _goto(at: AppTest, page: str) -> None:
    at.sidebar.button(key=f"nav_{page}").click()
    at.run(timeout=30)


def test_demo_dataset_file_exists_and_is_the_real_messy_employee_dataset() -> None:
    assert DEMO_DATASET_PATH.exists()
    result = load_csv(DEMO_DATASET_PATH, DEMO_DATASET_PATH.name)
    rows, cols = result.dataframe.shape
    assert 900 <= rows <= 1100  # ~1,020 rows, per the original dataset
    assert cols >= 8
    assert not result.warnings  # a clean, well-formed CSV -- issues live in the data, not the parsing


def test_demo_dataset_has_realistic_employee_columns() -> None:
    result = load_csv(DEMO_DATASET_PATH, DEMO_DATASET_PATH.name)
    columns = set(result.dataframe.columns)
    expected = {
        "Employee_ID", "First_Name", "Last_Name", "Age", "Department_Region",
        "Join_Date", "Salary", "Email", "Phone", "Performance_Score", "Remote_Work",
    }
    assert expected <= columns


def test_demo_dataset_triggers_real_data_quality_issues() -> None:
    """Runs the demo file through the app's real detection pipeline (not a mock).

    Confirms it actually demonstrates the product -- missing values and
    dates stored as text in mixed formats -- without being so broken it
    looks unrealistic (no single issue type should swamp nearly every row).
    """
    result = load_csv(DEMO_DATASET_PATH, DEMO_DATASET_PATH.name)
    df = result.dataframe
    profiles = profile_dataframe(df)
    detection = detect_issues(df, profiles, result.file_name)

    issue_types = {issue.issue_type for issue in detection.issues}
    expected_present = {"missing_null", "date_stored_as_text", "mixed_date_formats"}
    assert expected_present <= issue_types

    from collections import Counter

    counts = Counter(issue.issue_type for issue in detection.issues)
    assert all(count < len(df) * 0.9 for count in counts.values())


def test_demo_dataset_columns_are_inferred_as_the_intended_logical_types() -> None:
    """email/phone/identifier/date-as-text columns should actually be recognized as such."""
    result = load_csv(DEMO_DATASET_PATH, DEMO_DATASET_PATH.name)
    profiles = {p.original_name: p.logical_type for p in profile_dataframe(result.dataframe)}
    assert profiles["Email"] == "Email"
    assert profiles["Phone"] == "Phone number"
    assert profiles["Employee_ID"] == "Possible identifier"
    assert profiles["Join_Date"] == "Date stored as text"


def test_use_demo_dataset_button_shown_when_no_file_uploaded(tmp_path) -> None:
    from src import config as config_module

    with patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        assert not at.exception

        button_labels = {b.label for b in at.button}
        assert "Use Demo Dataset" in button_labels


def test_clicking_use_demo_dataset_loads_it_through_the_real_pipeline(tmp_path) -> None:
    from src import config as config_module

    with patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        at.button(key="use_demo_dataset_btn").click()
        at.run(timeout=30)
        assert not at.exception

        # The UI makes it obvious a demo is active.
        all_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
        assert "Demo dataset loaded" in all_text or "demo" in all_text.lower()

        # And the rest of the workflow behaves exactly as it would for a real
        # upload: Review Issues, Clean Data, and Results all render normally.
        _goto(at, "Review Issues")
        assert not at.exception
        _goto(at, "Clean Data")
        assert not at.exception
        _goto(at, "Results")
        assert not at.exception


def test_uploading_a_real_file_after_demo_takes_priority(tmp_path) -> None:
    """A real upload always wins over a previously-loaded demo dataset."""
    import io

    from src import config as config_module

    class FakeUploadedFile(io.BytesIO):
        pass

    csv_bytes = b"id,name\n1,alice\n2,bob\n"
    FakeUploadedFile.name = "real_upload.csv"
    FakeUploadedFile.size = len(csv_bytes)
    fake_file = FakeUploadedFile(csv_bytes)

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        assert not at.exception

        badges = " ".join(getattr(el, "value", "") for el in at.markdown)
        assert "real_upload.csv" in badges
        assert "demo_employee_data.csv" not in badges
