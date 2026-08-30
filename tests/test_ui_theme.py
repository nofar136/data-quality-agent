"""Tests for src.ui_theme -- the Phase 8 shared design-system helpers.

Only the pure, Streamlit-independent functions are covered here (chart
theming, severity styling); page_header/inject_global_css call st.markdown
directly and are exercised indirectly by the AppTest suite instead.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from src.ui_theme import (
    BEFORE_AFTER_MAP,
    SEVERITY_COLORS,
    apply_chart_theme,
    severity_badge,
    style_severity_column,
)


def test_apply_chart_theme_sets_white_background_and_minimal_gridlines() -> None:
    fig = px.bar(pd.DataFrame({"x": ["a", "b"], "y": [1, 2]}), x="x", y="y")
    themed = apply_chart_theme(fig)
    assert themed.layout.plot_bgcolor == "#FFFFFF"
    assert themed.layout.paper_bgcolor == "#FFFFFF"
    assert themed.layout.xaxis.showgrid is False


def test_apply_chart_theme_can_hide_the_legend() -> None:
    fig = px.bar(pd.DataFrame({"x": ["a"], "y": [1]}), x="x", y="y")
    themed = apply_chart_theme(fig, show_legend=False)
    assert themed.layout.showlegend is False


def test_before_after_map_uses_distinct_colors_for_before_and_after() -> None:
    assert BEFORE_AFTER_MAP["Before"] != BEFORE_AFTER_MAP["After"]


def test_severity_colors_cover_all_four_levels() -> None:
    assert set(SEVERITY_COLORS) == {"Critical", "High", "Medium", "Low"}


def test_severity_badge_embeds_the_severity_text_and_its_color() -> None:
    html = severity_badge("Critical")
    assert "Critical" in html
    assert SEVERITY_COLORS["Critical"] in html


def test_severity_badge_falls_back_gracefully_for_unknown_severity() -> None:
    html = severity_badge("Unmapped")
    assert "Unmapped" in html


def test_style_severity_column_returns_a_styler_when_column_present() -> None:
    df = pd.DataFrame({"Severity": ["Critical", "Low"], "Count": [1, 2]})
    styler = style_severity_column(df)
    rendered = styler.to_html()
    assert "Critical" in rendered
    assert "Low" in rendered


def test_style_severity_column_is_a_no_op_when_column_missing() -> None:
    df = pd.DataFrame({"Other": [1, 2]})
    styler = style_severity_column(df)
    assert "Other" in styler.to_html()
