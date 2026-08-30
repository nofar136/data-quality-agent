"""AppTest coverage for the Phase 8.2 information-architecture consolidation.

The 8-page nav collapsed into 4 primary steps (Upload, Review Issues, Clean
Data, Results) + 2 secondary pages (Run History, About). These tests exist
to prove every piece of content that used to live on its own page is still
reachable from its new home, and that the old separate workflow stepper is
genuinely gone (the numbered sidebar nav is the only progress indicator now).
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

import src.ui_components as ui_components

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _goto(at: AppTest, page: str, fake_file: "io.BytesIO | None" = None) -> None:
    if fake_file is not None:
        fake_file.seek(0)
    at.sidebar.button(key=f"nav_{page}").click()
    at.run(timeout=30)


def _fake_file(csv_bytes: bytes, name: str = "phase8_2_test.csv") -> "io.BytesIO":
    class FakeUploadedFile(io.BytesIO):
        pass

    FakeUploadedFile.name = name
    FakeUploadedFile.size = len(csv_bytes)
    return FakeUploadedFile(csv_bytes)


def _messy_dataset_csv() -> bytes:
    rows = [
        "1,10,Eng", "2,20,eng", "3,,Eng", "4,40,Sales", "5,50,Sales",
        "6,60,Sales", "7,70,Eng", "8,80,Eng", "9,90,Eng", "10,,Eng",
    ]
    return ("id,score,dept\n" + "\n".join(rows) + "\n").encode()


def _all_markdown_text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def test_workflow_stepper_helper_no_longer_exists() -> None:
    """Phase 8.2 removed the separate stepper entirely -- the nav itself is the only indicator."""
    assert not hasattr(ui_components, "workflow_stepper")


def test_no_duplicate_workflow_indicator_on_a_page(tmp_path) -> None:
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

        # No leftover "Workflow" caption and no stepper CSS classes anywhere in the render.
        all_text = _all_markdown_text(at) + " ".join(c.value for c in at.caption)
        assert "dqa-stepper" not in all_text
        assert "dqa-step-" not in all_text


def test_upload_page_shows_dataset_overview_content(tmp_path) -> None:
    fake_file = _fake_file(_messy_dataset_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        fake_file.seek(0)
        at.run(timeout=30)  # auto-jumps to Upload after the first load
        assert not at.exception

        metric_labels = {m.label for m in at.metric}
        assert {"Rows", "Columns", "Missing Values", "Duplicate Rows"} <= metric_labels
        assert "Dataset Preview" in _all_markdown_text(at)


def test_review_issues_page_shows_profiling_and_issues_content(tmp_path) -> None:
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

        # Column Profiling content (Section A)
        assert any(sb.label == "Select a column to inspect" for sb in at.selectbox)
        assert "Review a Column" in _all_markdown_text(at)

        # Data Quality Issues content (Section B)
        metric_labels = {m.label for m in at.metric}
        assert {"Total Issues", "Columns Affected", "High / Critical", "Needs Review"} <= metric_labels
        assert "Explore Detected Issues" in _all_markdown_text(at)


def test_results_page_shows_dashboard_content(tmp_path) -> None:
    fake_file = _fake_file(_messy_dataset_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        _goto(at, "Results", fake_file)
        assert not at.exception
        assert "Cleaning Summary" in _all_markdown_text(at)
        assert "Before vs After Dashboard" in _all_markdown_text(at)
        # No cleaning done yet -- the attractive empty state, not a wall of empty charts.
        assert "Your cleaned dataset will appear here" in _all_markdown_text(at)


def test_secondary_nav_still_reaches_run_history_and_about(tmp_path) -> None:
    fake_file = _fake_file(_messy_dataset_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)

        _goto(at, "Run History", fake_file)
        assert not at.exception
        assert "Run History" in _all_markdown_text(at)

        _goto(at, "About", fake_file)
        assert not at.exception
        assert "About" in _all_markdown_text(at)
