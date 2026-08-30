"""Lightweight app-level smoke tests using Streamlit's official AppTest harness.

These exercise app.py's actual script execution (imports, sidebar, page
dispatch) without a browser -- complementary to the pure-function tests in
test_type_override.py and test_dashboard_data.py, which cover the
underlying logic in detail. Kept intentionally small: broad "does every
page render" coverage, not deep behavioral assertions.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from app import NAV_PAGES, PRIMARY_NAV_PAGES, SECONDARY_NAV_PAGES

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _goto(at: AppTest, page: str, fake_file: "io.BytesIO | None" = None) -> None:
    """Navigate via the button-based sidebar nav (Phase 8.1) instead of a radio group."""
    if fake_file is not None:
        fake_file.seek(0)
    at.sidebar.button(key=f"nav_{page}").click()
    at.run(timeout=30)


def test_tableau_page_no_longer_appears() -> None:
    assert "Tableau Exports" not in NAV_PAGES


def test_nav_pages_match_the_required_workflow() -> None:
    """Phase 8.2: the 8-page nav collapsed into 4 primary steps + 2 secondary pages."""
    assert PRIMARY_NAV_PAGES == ("Upload", "Review Issues", "Clean Data", "Results")
    assert SECONDARY_NAV_PAGES == ("Run History", "About")
    assert NAV_PAGES == PRIMARY_NAV_PAGES + SECONDARY_NAV_PAGES


def test_no_technical_page_names_in_primary_navigation() -> None:
    """Dataset Overview / Column Profiling / Data Quality Dashboard are capabilities, not nav items."""
    technical_names = {"Dataset Overview", "Column Profiling", "Data Quality Dashboard", "Data Quality Issues", "Data Cleaning"}
    assert technical_names.isdisjoint(PRIMARY_NAV_PAGES)
    assert technical_names.isdisjoint(SECONDARY_NAV_PAGES)


def test_app_runs_without_exceptions_with_no_file_uploaded() -> None:
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    assert not at.exception


def _fake_uploaded_csv() -> "io.BytesIO":
    csv_bytes = (
        b"id,amount,notes\n"
        + b"".join(f"{i},{i},fine\n".encode() for i in range(1, 20))
        + b"20,abc,N/A\n"
    )

    class FakeUploadedFile(io.BytesIO):
        name = "smoke_test.csv"
        size = len(csv_bytes)

    return FakeUploadedFile(csv_bytes)


def test_every_page_renders_without_exceptions_with_a_loaded_file(tmp_path) -> None:
    fake_file = _fake_uploaded_csv()

    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        assert not at.exception

        for page in NAV_PAGES[1:]:
            _goto(at, page, fake_file)
            assert not at.exception, f"page '{page}' raised: {list(at.exception)}"
