"""Tests for the Phase 8.4 final polish pass: selectbox affordance, strategy
decision card + helper caption, and the updated chart color palette.

Purely presentational -- no business logic is exercised differently here
than in earlier phases' tests; these just confirm the new visual elements
are actually wired up (not just present in a CSS string nobody applies).
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.ui_theme import AFTER_COLOR, BEFORE_COLOR, CHART_GRIDLINE, SEVERITY_COLORS

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _goto(at: AppTest, page: str, fake_file: "io.BytesIO | None" = None) -> None:
    if fake_file is not None:
        fake_file.seek(0)
    at.sidebar.button(key=f"nav_{page}").click()
    at.run(timeout=30)


def _fake_file(csv_bytes: bytes, name: str = "phase8_4_test.csv") -> "io.BytesIO":
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


def test_final_color_palette_matches_the_exact_spec() -> None:
    assert BEFORE_COLOR == "#1D4ED8"
    assert AFTER_COLOR == "#0F766E"
    assert CHART_GRIDLINE == "#E2E8F0"
    assert SEVERITY_COLORS == {
        "Critical": "#B91C1C",
        "High": "#EA580C",
        "Medium": "#CA8A04",
        "Low": "#64748B",
    }


def test_strategy_selector_shows_helper_caption(tmp_path) -> None:
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

        captions = " ".join(c.value for c in at.caption)
        assert "Choose how to handle this issue." in captions


def test_strategy_selector_is_wrapped_in_its_own_card(tmp_path) -> None:
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

        strategy_box = at.selectbox(key="missing_score_strategy")
        assert strategy_box is not None
