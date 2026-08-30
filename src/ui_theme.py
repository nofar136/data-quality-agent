"""Centralized visual design system for the Streamlit UI (Phase 8 / 8.1).

Colors, the shared Plotly chart palette, global CSS, and small presentation
helpers live here so app.py's page functions stay focused on layout and
data wiring -- no design decision (a color, a spacing value, a chart
default) is duplicated or redefined page by page. Nothing in this module
reads application data or makes a business decision; it only renders what
it is handed. See ``.streamlit/config.toml`` for the native Streamlit theme
(app background/sidebar background/primary/text colors), which this module
complements with page headers, KPI cards, chart theming, and severity color.

CSS selectors here are verified against the installed Streamlit build's own
source (``data-testid`` attributes it renders), not guessed -- see the
module-level comments next to each rule for what they target and why that
target is stable (a documented testid, or the ``st-key-*`` class Streamlit
attaches to any element/container given an explicit ``key=``).
"""

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Color system --------------------------------------------------------------------

NAVY = "#172033"
PRIMARY = "#3563E9"
PRIMARY_SOFT = "#EAF0FF"
SUCCESS = "#18A999"
SUCCESS_SOFT = "#E9F8F5"
WARNING = "#E9A23B"
WARNING_SOFT = "#FFF6E5"
DANGER = "#D9534F"
DANGER_SOFT = "#FDECEC"
APP_BACKGROUND = "#F6F8FB"  # very light blue-gray -- the app canvas, not card surfaces
CARD = "#FFFFFF"
TEXT_PRIMARY = "#1F2937"
TEXT_SECONDARY = "#6B7280"
BORDER = "#E5E7EB"

# Sidebar-only palette (the sidebar is a dark navy brand surface; the main
# workspace stays light -- see .streamlit/config.toml, which is NOT changed
# to dark, plus the scoped `section[data-testid="stSidebar"]` CSS below).
SIDEBAR_BACKGROUND = NAVY
SIDEBAR_TEXT_PRIMARY = "#FFFFFF"
SIDEBAR_TEXT_SECONDARY = "#B9C2D0"

# Semantic roles -- the same concept always uses the same color everywhere.
# Chart-specific shades (Phase 8.4 polish): stronger contrast than the
# softer PRIMARY/SUCCESS used for badges/accents elsewhere, per explicit
# design direction -- Before is always blue, After is always green/teal.
BEFORE_COLOR = "#1D4ED8"  # original / before cleaning
AFTER_COLOR = "#0F766E"  # cleaned / after -- and positive improvement generally
NEUTRAL_COLOR = "#94A3B8"
CHART_GRIDLINE = "#E2E8F0"

BEFORE_AFTER_MAP: dict[str, str] = {"Before": BEFORE_COLOR, "After": AFTER_COLOR}

SEVERITY_COLORS: dict[str, str] = {
    "Critical": "#B91C1C",
    "High": "#EA580C",
    "Medium": "#CA8A04",
    "Low": "#64748B",
}
SEVERITY_BACKGROUNDS: dict[str, str] = {
    "Critical": DANGER_SOFT,
    "High": "#FCEEE9",
    "Medium": WARNING_SOFT,
    "Low": "#EEF1F5",
}

_ACCENT_COLORS: dict[str, str] = {"primary": PRIMARY, "success": SUCCESS, "warning": WARNING, "danger": DANGER, "neutral": NEUTRAL_COLOR}

CHART_FONT = "'Segoe UI', -apple-system, Helvetica, Arial, sans-serif"


# --- Chart theming ---------------------------------------------------------------------


def apply_chart_theme(fig: go.Figure, *, height: Optional[int] = None, show_legend: bool = True) -> go.Figure:
    """Apply the shared, minimal chart look: white background, light gridlines, compact legend.

    Never assigns data colors itself -- callers pass an explicit
    ``color_discrete_map`` (see BEFORE_AFTER_MAP / SEVERITY_COLORS) so a
    concept never gets a different color on different charts.
    """
    fig.update_layout(
        template="plotly_white",
        font=dict(family=CHART_FONT, size=13, color=TEXT_PRIMARY),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None) if show_legend else dict(visible=False),
        showlegend=show_legend,
        plot_bgcolor=CARD,
        paper_bgcolor=CARD,
        hoverlabel=dict(bgcolor=CARD, font_color=TEXT_PRIMARY, bordercolor=BORDER),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=CHART_GRIDLINE)
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRIDLINE, zeroline=False)
    if height:
        fig.update_layout(height=height)
    return fig


# --- Table styling -----------------------------------------------------------------------


def style_severity_column(df: pd.DataFrame, column: str = "Severity"):
    """Color-code a severity column consistently with the rest of the app.

    Returns a pandas Styler (renderable directly by st.dataframe) so
    severity is never the *only* signal -- the text itself still reads
    "Critical" / "High" / etc.; color is a reinforcing cue, not a
    replacement for the label.
    """
    if column not in df.columns:
        return df.style

    def _style(value: object) -> str:
        color = SEVERITY_COLORS.get(str(value), TEXT_SECONDARY)
        bg = SEVERITY_BACKGROUNDS.get(str(value), "#F3F4F6")
        return f"color: {color}; background-color: {bg}; font-weight: 600; border-radius: 4px;"

    return df.style.map(_style, subset=[column])


# --- Reusable page chrome ------------------------------------------------------------------


def page_header(title: str, subtitle: str, badge: Optional[str] = None) -> None:
    """Render the shared page-header pattern: a short title + one concise subtitle.

    ``badge`` is an optional small contextual pill (e.g. the loaded file
    name) shown next to the title -- never a paragraph, just a label.
    """
    badge_html = f'<span class="dqa-header-badge">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="dqa-page-title">{title}{badge_html}</div><div class="dqa-page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def severity_badge(severity: str) -> str:
    """Inline HTML badge for a severity value, for use inside st.markdown(unsafe_allow_html=True).

    Only ever called with one of the four fixed severity labels the app
    itself assigns (never raw user/file data), so this never renders
    untrusted input as HTML.
    """
    color = SEVERITY_COLORS.get(severity, TEXT_SECONDARY)
    bg = SEVERITY_BACKGROUNDS.get(severity, "#F3F4F6")
    return f'<span class="dqa-badge" style="color:{color};background-color:{bg};">{severity}</span>'


def plain_badge(text: str) -> str:
    """Inline HTML pill for a neutral label (e.g. a technology name) -- no severity semantics."""
    return f'<span class="dqa-badge" style="color:{TEXT_SECONDARY};background-color:#F3F4F6;">{text}</span>'


AccentName = Literal["primary", "success", "warning", "danger", "neutral"]


def kpi_card_row(cards: list[dict]) -> None:
    """Render a row of polished, custom KPI cards as one HTML block.

    Used where st.metric's fixed look isn't distinctive enough (the
    Dashboard's headline numbers) -- st.metric (restyled by the global CSS
    below) covers every other KPI in the app, so this stays reserved for
    the one page meant to be the visual highlight.

    Each card dict: {"label": str, "value": str, "delta": Optional[str],
    "accent": one of AccentName}. All values are caller-formatted strings
    (numbers already rendered, e.g. "84.2 / 100") -- this function only lays
    them out, it never computes or formats a metric itself. Callers only
    ever pass internally-computed, already-formatted numbers/labels here,
    never raw file content, so the HTML interpolation below is safe.
    """
    cells = []
    for card in cards:
        accent = _ACCENT_COLORS.get(card.get("accent", "primary"), PRIMARY)
        delta = card.get("delta")
        delta_html = f'<div class="dqa-kpi-delta" style="color:{accent}">{delta}</div>' if delta else ""
        cells.append(
            f'<div class="dqa-kpi-card">'
            f'<div class="dqa-kpi-accent" style="background:{accent}"></div>'
            f'<div class="dqa-kpi-label">{card["label"]}</div>'
            f'<div class="dqa-kpi-value">{card["value"]}</div>'
            f"{delta_html}"
            f"</div>"
        )
    st.markdown(f'<div class="dqa-kpi-row">{"".join(cells)}</div>', unsafe_allow_html=True)


# --- Global CSS ----------------------------------------------------------------------------
#
# Selector notes (verified against the installed streamlit build's JS source):
#   - `[class*="st-key-card"]`      -- Streamlit adds a `st-key-<key>` class to
#     any element/container given `key=`; every card-style container in this
#     app uses a key starting with "card", so one rule styles them all.
#   - `stBaseButton-primary/secondary` -- the literal `data-testid` React
#     renders on every button, one per `type=`.
#   - `stSidebar`, `stSidebarUserContent`, `stMetric*`, `stExpander` are
#     Streamlit's own long-standing testids for those components.

_CSS = f"""
<style>
.block-container {{
    max-width: 1280px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}}
.dqa-page-title {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {NAVY};
    line-height: 1.3;
    margin-bottom: 0.15rem;
}}
.dqa-header-badge {{
    display: inline-block;
    margin-left: 0.6rem;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    background: {PRIMARY_SOFT};
    color: {PRIMARY};
    font-size: 0.72rem;
    font-weight: 600;
    vertical-align: middle;
}}
.dqa-page-subtitle {{
    font-size: 0.92rem;
    color: {TEXT_SECONDARY};
    margin-bottom: 1rem;
}}

/* --- Sidebar: dark navy brand surface, tightened spacing, real nav buttons ---
   Scoped entirely to section[data-testid="stSidebar"] -- the main workspace
   theme (.streamlit/config.toml) is untouched and stays light. The file
   uploader's own drop-zone widget is explicitly excepted below so its
   (light) internal contents keep normal dark-on-light contrast. */
section[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BACKGROUND};
    border-right: none;
}}
div[data-testid="stSidebarUserContent"] {{
    padding-top: 1.1rem;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {{
    color: {SIDEBAR_TEXT_SECONDARY};
}}
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] * {{
    color: {TEXT_PRIMARY} !important;
}}
.dqa-sidebar-title {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {SIDEBAR_TEXT_PRIMARY} !important;
    margin-bottom: 0;
    line-height: 1.2;
}}
.dqa-sidebar-subtitle {{
    font-size: 0.78rem;
    color: {SIDEBAR_TEXT_SECONDARY} !important;
    margin-bottom: 0.75rem;
}}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{
    gap: 0.35rem;
}}
.dqa-nav-divider {{
    border-top: 1px solid rgba(255, 255, 255, 0.12);
    margin: 0.75rem 0;
}}
div[class*="st-key-nav_"] button[data-testid^="stBaseButton"] {{
    justify-content: flex-start !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 0.45rem 0.75rem !important;
    width: 100%;
}}
div[class*="st-key-nav_"] button[data-testid^="stBaseButton"] * {{
    color: inherit !important;
}}
div[class*="st-key-nav_"] button[data-testid="stBaseButton-secondary"] {{
    background: transparent !important;
    border-color: transparent !important;
    color: {SIDEBAR_TEXT_SECONDARY} !important;
}}
div[class*="st-key-nav_"] button[data-testid="stBaseButton-secondary"]:hover {{
    background: rgba(255, 255, 255, 0.08) !important;
    color: {SIDEBAR_TEXT_PRIMARY} !important;
}}
div[class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] {{
    background: {PRIMARY} !important;
    border-color: transparent !important;
    color: {SIDEBAR_TEXT_PRIMARY} !important;
    font-weight: 700 !important;
}}

/* --- Cards: any container keyed "card*" (see key= call sites) --- */
div[class*="st-key-card"] {{
    background: {CARD};
    border: 1px solid rgba(23, 32, 51, 0.06);
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(23, 32, 51, 0.06);
    padding: 0.9rem 1.1rem;
}}

/* --- Upload page empty-state hero: the one prominent central call to action --- */
div[class*="st-key-card_upload_hero"] {{
    padding: 2.5rem 1.5rem;
    text-align: center;
}}
.dqa-upload-hero-icon {{
    font-size: 2rem;
    color: {PRIMARY};
    margin-bottom: 0.5rem;
}}
.dqa-upload-hero-title {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {NAVY};
    margin-bottom: 0.25rem;
}}
.dqa-upload-hero-subtitle {{
    font-size: 0.9rem;
    color: {TEXT_SECONDARY};
}}

/* --- Selectbox (single-choice dropdowns): make the control unmistakably
   interactive -- stronger border, white surface, rounded corners, a clear
   focus ring, and a bigger, colored chevron. Selectors use stable ARIA
   roles (role="group"/"combobox", aria-haspopup) rather than Streamlit's
   auto-generated emotion class names, so they hold up across versions. */
div[data-testid="stSelectbox"] div[role="group"] {{
    background: {CARD} !important;
    border: 1.5px solid #C7D0DC !important;
    border-radius: 10px !important;
    min-height: 2.75rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
div[data-testid="stSelectbox"] div[role="group"]:hover {{
    border-color: #94A3B8 !important;
}}
div[data-testid="stSelectbox"] div[role="group"]:focus-within {{
    border-color: {PRIMARY} !important;
    box-shadow: 0 0 0 3px {PRIMARY_SOFT};
}}
div[data-testid="stSelectbox"] input[role="combobox"] {{
    padding: 0.6rem 0.5rem 0.6rem 0.9rem !important;
    font-weight: 500;
    color: {TEXT_PRIMARY} !important;
    cursor: pointer;
}}
div[data-testid="stSelectbox"] button[aria-haspopup="listbox"] {{
    padding-right: 0.85rem;
}}
div[data-testid="stSelectbox"] button[aria-haspopup="listbox"] svg {{
    width: 1.4rem;
    height: 1.4rem;
    color: {PRIMARY};
}}

/* --- KPI metric cards (st.metric, used outside the Dashboard) --- */
div[data-testid="stMetric"] {{
    background-color: {CARD};
    border: 1px solid rgba(23, 32, 51, 0.06);
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(23, 32, 51, 0.06);
    padding: 0.9rem 1.1rem 0.75rem 1.1rem;
}}
div[data-testid="stMetricLabel"] {{ color: {TEXT_SECONDARY}; }}
div[data-testid="stMetricValue"] {{
    color: {NAVY};
    font-size: 1.35rem;
    line-height: 1.3;
}}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {{
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
}}

/* --- Custom KPI cards (Dashboard headline numbers) --- */
.dqa-kpi-row {{ display: flex; flex-wrap: wrap; gap: 0.85rem; margin-bottom: 0.5rem; }}
.dqa-kpi-card {{
    position: relative;
    flex: 1 1 160px;
    background: {CARD};
    border: 1px solid rgba(23, 32, 51, 0.06);
    border-radius: 14px;
    box-shadow: 0 2px 6px rgba(23, 32, 51, 0.08);
    padding: 1rem 1.1rem 0.85rem 1.1rem;
    overflow: hidden;
}}
.dqa-kpi-accent {{ position: absolute; top: 0; left: 0; width: 100%; height: 3px; }}
.dqa-kpi-label {{ font-size: 0.78rem; color: {TEXT_SECONDARY}; margin-bottom: 0.2rem; }}
.dqa-kpi-value {{ font-size: 1.5rem; font-weight: 700; color: {NAVY}; line-height: 1.2; }}
.dqa-kpi-delta {{ font-size: 0.8rem; font-weight: 600; margin-top: 0.15rem; }}

.dqa-badge {{
    display: inline-block;
    padding: 0.1rem 0.55rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}}

div[data-testid="stExpander"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {CARD};
}}
hr {{ border-color: {BORDER}; }}
</style>
"""


def inject_global_css() -> None:
    """Inject the app-wide CSS once per render. Idempotent -- safe to call every rerun."""
    st.markdown(_CSS, unsafe_allow_html=True)
