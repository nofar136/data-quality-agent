"""AppTest coverage for the Phase 8.1 UX pass (button nav, stepper, "summary first" tables).

Complements test_phase8_ui.py (Phase 8's coverage, still valid) with checks
specific to what changed in 8.1: zero-impact safe fixes staying out of the
primary view, and the new expandable-table "view more" pattern actually
revealing additional rows rather than just rendering a checkbox that does
nothing.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _goto(at: AppTest, page: str, fake_file: "io.BytesIO | None" = None) -> None:
    if fake_file is not None:
        fake_file.seek(0)
    at.sidebar.button(key=f"nav_{page}").click()
    at.run(timeout=30)


def _fake_file(csv_bytes: bytes, name: str = "phase8_1_test.csv") -> "io.BytesIO":
    class FakeUploadedFile(io.BytesIO):
        pass

    FakeUploadedFile.name = name
    FakeUploadedFile.size = len(csv_bytes)
    return FakeUploadedFile(csv_bytes)


def _already_clean_csv() -> bytes:
    # Snake_case column names, no missing values, no duplicates, no
    # text-stored dates/numbers -- every safe fix should be zero-impact.
    rows = [f"{i},{100 + i},active" for i in range(1, 21)]
    return ("id,score,status\n" + "\n".join(rows) + "\n").encode()


def test_zero_impact_safe_fixes_are_not_shown_as_active_cards(tmp_path) -> None:
    fake_file = _fake_file(_already_clean_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        _goto(at, "Clean Data", fake_file)
        assert not at.exception

        active_fix_checkboxes = [c for c in at.checkbox if c.key and c.key.startswith("fix_")]
        assert active_fix_checkboxes == []

        expanders = [e for e in at.expander if e.label.startswith("Other available checks")]
        assert len(expanders) == 1


def test_expandable_table_view_more_reveals_additional_rows(tmp_path) -> None:
    fake_file = _fake_file(_already_clean_csv())
    from src import config as config_module

    with (
        patch.object(st.sidebar, "file_uploader", return_value=fake_file),
        patch.object(config_module, "DATABASE_PATH", tmp_path / "db.sqlite"),
    ):
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        fake_file.seek(0)
        at.run(timeout=30)  # lands on Upload (auto-jump on first load; shows the dataset overview inline)
        assert not at.exception

        dataframes_before = at.dataframe
        assert len(dataframes_before) >= 1
        preview_rows_before = len(dataframes_before[0].value)
        assert preview_rows_before == 10  # QUICK_PREVIEW_ROWS-equivalent default for the dataset preview

        view_more = at.checkbox(key="preview_view_all")
        view_more.check()
        fake_file.seek(0)
        at.run(timeout=30)
        assert not at.exception

        dataframes_after = at.dataframe
        assert len(dataframes_after) == 2  # the 10-row preview stays, plus the newly revealed full table
        assert len(dataframes_after[0].value) == 10
        assert len(dataframes_after[1].value) == 20  # the full 20-row dataset is now shown
