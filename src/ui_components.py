"""Reusable, stateful UI building blocks (Phase 8.1 / 8.2).

Distinct from ``src/ui_theme.py`` (pure design tokens, CSS, and stateless
render helpers): everything here composes those tokens into a small
interactive piece of the app shell -- sidebar navigation reads/writes
``st.session_state`` and calls ``st.rerun()``, and the expandable-table
helper is a small stateful widget (a checkbox). Still presentation-only: no
business logic, no data mutation, nothing that touches profiling/cleaning/scoring.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def sidebar_nav(
    primary_pages: tuple[str, ...],
    secondary_pages: tuple[str, ...],
    current_page: str,
    completed_pages: set[str],
    *,
    key_prefix: str = "nav",
) -> str:
    """Render the sidebar navigation: a numbered primary workflow, then secondary pages below a divider.

    The sidebar navigation *is* the workflow indicator (Phase 8.2 removed
    the separate stepper) -- primary pages are numbered "01", "02", ... and
    show a checkmark once ``completed_pages`` says they're done; secondary
    pages (Run History, About) are plain, unnumbered buttons below a divider
    so they never compete visually with the four main steps.

    A plain ``st.radio`` renders as a native radio group with no reliable,
    stable way to CSS-style just the selected option (Streamlit's BaseWeb
    radio computes its selected-dot color from React state, not a fixed
    class or attribute). Buttons don't have that problem: the active page
    is simply the one rendered with ``type="primary"``, everything else
    ``type="secondary"`` -- both are first-class, themeable Streamlit APIs.

    Returns the page that should be active after this render (unchanged
    unless the user just clicked a different one, in which case this also
    triggers the rerun itself so callers can treat the return value as final).
    """
    selected = current_page

    for idx, page in enumerate(primary_pages, start=1):
        is_active = page == current_page
        checkmark = " ✓" if (page in completed_pages and not is_active) else ""
        label = f"{idx:02d}  {page}{checkmark}"
        if st.sidebar.button(label, key=f"{key_prefix}_{page}", type="primary" if is_active else "secondary", width="stretch"):
            selected = page

    st.sidebar.markdown('<div class="dqa-nav-divider"></div>', unsafe_allow_html=True)

    for page in secondary_pages:
        is_active = page == current_page
        if st.sidebar.button(page, key=f"{key_prefix}_{page}", type="primary" if is_active else "secondary", width="stretch"):
            selected = page

    if selected != current_page:
        st.session_state["nav_page"] = selected
        st.rerun()
    return selected


def expandable_table(
    df: pd.DataFrame, *, preview_rows: int, key: str, max_full_rows: int = 500, empty_message: str = "No rows.", **dataframe_kwargs
) -> None:
    """Render a small preview of a table, with a checkbox to reveal more.

    The "summary first, details on demand" pattern used throughout the app:
    a table with more than ``preview_rows`` rows shows only the first
    ``preview_rows`` until the user explicitly asks for more, instead of
    dumping a potentially huge table on the page by default. Even then,
    display is capped at ``max_full_rows`` -- this is a UI convenience, not
    a data limit; every download still contains the complete data.
    """
    if df.empty:
        st.caption(empty_message)
        return
    st.dataframe(df.head(preview_rows), hide_index=True, **dataframe_kwargs)
    if len(df) > preview_rows and st.checkbox(f"View more rows ({len(df):,} total)", key=key):
        if len(df) > max_full_rows:
            st.caption(f"Showing the first {max_full_rows:,} of {len(df):,} rows.")
        st.dataframe(df.head(max_full_rows), hide_index=True, **dataframe_kwargs)
