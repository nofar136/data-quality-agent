"""AppTest coverage for the Phase 8 visual/UX redesign.

Phase 8 changed no business logic -- these tests exist to catch UI wiring
regressions the redesign could introduce (a broken container/expander
nesting, a chart that errors on an edge-case DataFrame, a widget key that
silently stopped working) that the pure src/ unit tests can't see, since
those never touch Streamlit rendering at all.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from app import NAV_PAGES

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _goto(at: AppTest, page: str, fake_file: "io.BytesIO | None" = None) -> None:
    """Navigate via the button-based sidebar nav (Phase 8.1) instead of a radio group."""
    if fake_file is not None:
        fake_file.seek(0)
    at.sidebar.button(key=f"nav_{page}").click()
    at.run(timeout=30)


def _fake_file(csv_bytes: bytes, name: str = "phase8_test.csv") -> "io.BytesIO":
    class FakeUploadedFile(io.BytesIO):
        pass

    FakeUploadedFile.name = name
    FakeUploadedFile.size = len(csv_bytes)
    return FakeUploadedFile(csv_bytes)


def _clean_dataset_csv() -> bytes:
    # No missing values, no duplicates, no ambiguous types -- zero issues,
    # exercising every chart/table's "empty" branch.
    rows = [f"{i},{100 + i},active" for i in range(1, 31)]
    return ("id,score,status\n" + "\n".join(rows) + "\n").encode()


def _single_category_dataset_csv() -> bytes:
    # Only missing values (Missing Value category) -- no type, text, or
    # numeric-category issues -- exercising the "one category" chart path.
    rows = ["1,10", "2,", "3,30", "4,", "5,50", "6,60", "7,70", "8,80", "9,90", "10,100"]
    return ("id,amount\n" + "\n".join(rows) + "\n").encode()


def test_zero_issues_dataset_renders_every_page_without_exceptions(tmp_path) -> None:
    fake_file = _fake_file(_clean_dataset_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        for page in NAV_PAGES[1:]:
            _goto(at, page, fake_file)
            assert not at.exception, f"page '{page}' raised on a zero-issue dataset: {list(at.exception)}"


def test_single_category_dataset_renders_dashboard_without_exceptions(tmp_path) -> None:
    fake_file = _fake_file(_single_category_dataset_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        _goto(at, "Results", fake_file)
        assert not at.exception


def test_type_override_control_still_functional_after_redesign(tmp_path) -> None:
    fake_file = _fake_file(_clean_dataset_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        _goto(at, "Review Issues", fake_file)
        assert not at.exception

        inspect_box = next(sb for sb in at.selectbox if sb.label == "Select a column to inspect")
        inspect_box.select("status")
        fake_file.seek(0)
        at.run(timeout=30)
        assert not at.exception

        override_box = at.selectbox(key="type_override_select_status")
        non_default = next(label for label in override_box.options if label != override_box.value)
        override_box.select(non_default)
        fake_file.seek(0)
        at.run(timeout=30)
        assert not at.exception
        assert at.session_state["type_overrides"]["status"] == non_default


def test_dashboard_renders_after_a_guided_decision_is_applied(tmp_path) -> None:
    fake_file = _fake_file(_single_category_dataset_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        _goto(at, "Clean Data", fake_file)

        strategy_box = at.selectbox(key="missing_amount_strategy")
        strategy_box.select("Replace with 0")
        fake_file.seek(0)
        at.run(timeout=30)

        at.button(key="missing_amount_apply").click()
        fake_file.seek(0)
        at.run(timeout=30)
        assert not at.exception

        _goto(at, "Results", fake_file)
        assert not at.exception


def test_download_controls_present_after_cleaning(tmp_path) -> None:
    fake_file = _fake_file(_single_category_dataset_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        _goto(at, "Clean Data", fake_file)

        strategy_box = at.selectbox(key="missing_amount_strategy")
        strategy_box.select("Replace with 0")
        fake_file.seek(0)
        at.run(timeout=30)

        at.button(key="missing_amount_apply").click()
        fake_file.seek(0)
        at.run(timeout=30)
        assert not at.exception

        _goto(at, "Results", fake_file)  # downloads live on Results, not Clean Data (Phase 8.2)
        download_labels = {b.label for b in at.download_button}
        assert "Download Cleaned CSV" in download_labels
        assert "Decision Log" in download_labels
        assert len(download_labels) >= 6
