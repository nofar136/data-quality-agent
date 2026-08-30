"""Generic AI Data Quality Agent - Streamlit entry point.

Phase 1 scope: file upload, robust parsing (CSV/Excel), dataset preview,
and basic dataset info.
Phase 2 scope: generic schema inference and column profiling.
Phase 3 scope: generic data quality issue detection and review (detection
only -- no cleaning yet).
Phase 4 scope: safe cleaning with explicit user approval, before/after
comparison, and a full audit log.
Phase 5 scope: transparent data quality scoring, SQLite persistence, and
run history/comparison.
Phase 6 scope: a sidebar-navigated workflow and a polished, portfolio-ready
presentation.
Phase 7A scope: human-in-the-loop logical type override (Column Profiling
page) and an interactive Data Quality Dashboard (Plotly). Tableau exports
were removed from the product direction -- see src/type_override.py and
src/dashboard_data.py for the new business logic; app.py only renders it.
Phase 7B scope: guided, human-in-the-loop cleaning. The Data Cleaning page
gained a "Guided issue review" workflow -- issues grouped by column + issue
type, each offering only the remediation strategies that make sense for its
effective logical type (src/cleaning_strategies.py), always previewed
before an explicit "Apply Cleaning Decision" click, with every decision
(including "keep" / "do not clean") logged separately from the audit log of
actual value changes (src/models.py:CleaningDecisionLogEntry).
Phase 8 scope: visual/UX redesign only -- no business logic changed. A
shared design system (colors, chart theme, page-header/severity-styling
helpers) lives in src/ui_theme.py; app.py only calls into it. See
.streamlit/config.toml for the native Streamlit color theme.
Phase 8.1 scope: a deeper visual/UX pass, still no business logic changed --
button-based sidebar navigation and a visual workflow stepper
(src/ui_components.py), aggressively shortened on-page text (detail moved
into expanders), and "summary first, details on demand" tables everywhere
via src.ui_components.expandable_table.
Phase 8.2 scope: information-architecture consolidation, still no business
logic changed. The 8-page nav collapsed into a 4-step primary workflow --
Upload (was Upload Data + Dataset Overview), Review Issues (was Column
Profiling + Data Quality Issues), Clean Data (was Data Cleaning), Results
(was Data Quality Dashboard + downloads) -- with Run History/About demoted
to a secondary nav group. The separate workflow stepper was removed; the
numbered, checkmarked sidebar nav (src.ui_components.sidebar_nav) is now
the only workflow indicator. Every underlying render function from 8.1
still exists and is simply called from a new page in a new order.
"""

from __future__ import annotations

import io
import uuid
from collections import Counter
from dataclasses import replace as dataclasses_replace
from datetime import date, datetime, timezone
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.cleaning_engine import SAFE_FIX_DEFINITIONS, CleaningResult, apply_selected_fixes, apply_value_replacements, preview_fixes
from src.cleaning_strategies import (
    TYPE_CONFIRMATION_NOTE,
    CleaningStrategyOption,
    StrategyCatalogResult,
    get_categorical_variant_strategies,
    get_missing_value_strategies,
    get_negative_value_decision_options,
    get_negative_value_treatment_strategies,
    get_outlier_strategies,
)
from src.config import (
    COMPONENT_WEIGHTS,
    DATABASE_PATH,
    DEFAULT_MAX_ISSUES_DISPLAYED,
    DEMO_DATASET_PATH,
    EXCEL_EXTENSIONS,
    MAX_SAMPLE_ROWS_TO_INSPECT,
    TYPE_OVERRIDE_CONFIDENCE_WARNING_THRESHOLD,
)
from src.dashboard_data import (
    cleaning_impact_summary,
    compute_kpis,
    issues_by_category_comparison,
    issues_by_severity_comparison,
    issues_total_comparison,
    most_problematic_columns,
    quality_component_comparison,
    unresolved_issues_summary,
)
from src.database import AppliedFixSummary, DatabaseError, RunBundle, get_run_detail, get_run_history, init_db, save_run
from src.file_loader import FileLoadError, LoadedDataset, get_excel_sheet_names, load_csv, load_excel
from src.issue_detector import ISSUE_TYPE_EXPLANATIONS
from src.issue_grouping import IssueGroup, build_issue_groups
from src.models import AuditLogEntry, CleaningDecisionLogEntry, DetectionSummary, Issue
from src.profiler import ColumnProfile, numeric_values, profile_dataframe
from src.rule_engine import detect_issues
from src.schema_inference import LogicalType
from src.scoring import QualityScore, calculate_quality_score
from src.type_override import OVERRIDE_LABELS, USE_DETECTED_LABEL, apply_type_overrides, is_low_confidence, make_override_record
from src.ui_components import expandable_table, sidebar_nav
from src.ui_theme import (
    AFTER_COLOR,
    BEFORE_AFTER_MAP,
    apply_chart_theme,
    inject_global_css,
    kpi_card_row,
    page_header,
    plain_badge,
    severity_badge,
    style_severity_column,
)
from src.utils import human_readable_size

PRIMARY_NAV_PAGES: tuple[str, ...] = ("Upload", "Review Issues", "Clean Data", "Results")
SECONDARY_NAV_PAGES: tuple[str, ...] = ("Run History", "About")
NAV_PAGES: tuple[str, ...] = PRIMARY_NAV_PAGES + SECONDARY_NAV_PAGES

GLOSSARY: dict[str, str] = {
    "Data profiling": "Automatically summarizing a dataset's shape and each column's statistics "
    "(missing values, uniqueness, distribution) without knowing anything about it in advance.",
    "Logical type": "What a column actually represents (Email, Date, Possible identifier, ...), inferred "
    "from its values -- independent of its pandas storage dtype and never assumed from its name alone.",
    "Type override": "A user's explicit correction to a column's inferred logical type, scoped to the "
    "current session -- it changes which checks run, never the underlying data, and is fully auditable.",
    "Data quality issue": "A specific, explainable problem found in the data (e.g. a missing value, an "
    "inconsistent date format), always tagged with a category, severity, and confidence.",
    "Safe fix": "An automatic correction applied only when it cannot lose information or introduce "
    "ambiguity (e.g. trimming whitespace) -- never filling in missing values or guessing at intent.",
    "Audit log": "A complete record of every change a safe fix made: the original value, the new value, "
    "which rule made the change, and why -- so cleaning is always traceable back to the source data.",
    "Data quality score": "A transparent 0-100 score built from five weighted components (Completeness, "
    "Uniqueness, Validity, Consistency, Structural Quality), each computed from a documented formula.",
    "Run history": "Every profiling/cleaning session explicitly saved to SQLite, so quality scores and "
    "issue counts can be compared across datasets and over time, even after restarting the app.",
}

_SCORE_METHODOLOGY_MARKDOWN = """
The overall score is a **weighted sum of five component scores**, each 0-100:

`overall = sum(component_score x component_weight) / 100`

Every component is computed the same way: `score = 100 - penalty`, where
`penalty = min(issue_count / denominator x 100, 100)`. The denominator is
chosen per component so the ratio is meaningful (see below) -- nothing is
ever divided by "total cells" indiscriminately.

| Component | What it measures | Denominator |
|---|---|---|
| Completeness | Missing, blank, whitespace-only, or placeholder cells | Total cells |
| Uniqueness | Rows that are exact duplicates or share a duplicate identifier value (counted once each, even if both apply) | Total rows |
| Validity | Cells that are outright invalid: failed conversions, unparseable dates, infinite values | Total non-missing cells |
| Consistency | Formatting problems: whitespace, capitalization, near-duplicate categories, mixed formats | Total non-missing cells |
| Structural Quality | Empty rows/columns, duplicate/inconsistent column names, columns stored in the wrong type | Total rows + columns |

Every detected issue counts toward **at most one** component -- the mapping
in `src/config.py` (`*_ISSUE_TYPES`) is disjoint by construction, so nothing
is penalized twice. Statistical outliers, constant columns, and unusual
(but plausible) dates are informational only and never affect the score,
since Phase 3 explicitly does not treat them as errors.
"""


def _get_or_create_run_id(cleaning_result: CleaningResult | None, dataset_key: str) -> str:
    """Return a stable run_id for the current session.

    Re-using ``cleaning_result.run_id`` when cleaning was performed means a
    fresh "Apply Selected Fixes" click always produces a genuinely new run.
    Without cleaning, the id is stable for as long as the same file stays
    loaded, so re-clicking "Save" without changing anything correctly hits
    the database's duplicate-run protection instead of silently saving a copy.
    """
    if cleaning_result is not None:
        return cleaning_result.run_id
    session_key = f"original_run_id::{dataset_key}"
    if session_key not in st.session_state:
        st.session_state[session_key] = str(uuid.uuid4())
    return st.session_state[session_key]

_SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]

QUICK_PREVIEW_ROWS: int = 10  # "summary first" default row count for secondary/diagnostic tables

# Human-facing labels for internal issue_type slugs (display only -- the raw
# slug is still what src/ uses internally, in SQLite, and in audit/decision
# log entries; this dict never changes detection, storage, or export shape,
# it only decides what a person reads on screen).
ISSUE_TYPE_LABELS: dict[str, str] = {
    "blank_string": "Blank string",
    "column_name_whitespace": "Column name has whitespace",
    "date_stored_as_text": "Date stored as text",
    "empty_column": "Empty column",
    "empty_row": "Empty row",
    "exact_duplicate_row": "Duplicate row",
    "identifier_duplicate_value": "Duplicate identifier",
    "identifier_inconsistent_format": "Inconsistent identifier format",
    "identifier_missing_values": "Missing identifier",
    "inconsistent_capitalization": "Inconsistent capitalization",
    "inconsistent_column_name_formatting": "Inconsistent column name formatting",
    "infinite_value": "Infinite value",
    "invalid_date": "Invalid date",
    "leading_trailing_whitespace": "Leading/trailing whitespace",
    "missing_null": "Missing values",
    "missing_placeholder": "Missing value placeholder",
    "mixed_data_types": "Mixed data types",
    "mixed_date_formats": "Mixed date formats",
    "negative_value": "Negative value",
    "non_printable_characters": "Non-printable characters",
    "numeric_format_inconsistency": "Inconsistent numeric format",
    "numeric_stored_as_text": "Number stored as text",
    "possible_outlier": "Possible outlier",
    "repeated_internal_spaces": "Repeated spaces",
    "similar_category_values": "Similar category values",
    "suspiciously_constant_column": "Suspiciously constant column",
    "unexpected_text_in_numeric_column": "Unexpected text in numeric column",
    "unusually_future_date": "Unusually future date",
    "unusually_old_date": "Unusually old date",
    "value_fails_type_conversion": "Value fails type conversion",
    "whitespace_only_string": "Whitespace-only value",
}


def humanize_issue_type(issue_type: str) -> str:
    """Human-facing label for an issue_type slug, for any prominent on-screen use.

    Falls back to a generic "underscores -> spaces, capitalized" rendering
    for any issue type not in ISSUE_TYPE_LABELS, so a future rule never
    accidentally surfaces a raw slug just because this dict wasn't updated.
    """
    return ISSUE_TYPE_LABELS.get(issue_type, issue_type.replace("_", " ").capitalize())

_SEVERITY_EXPLANATION = """
Severity is always assigned by a fixed, deterministic rule -- never randomly:

- **Ratio-based** (missing values, duplicate rows, empty rows): severity climbs from
  Low to Critical as the *proportion* of affected rows in that column/dataset grows.
- **Fixed-by-type** (e.g. an empty column, a mixed-type column, an infinite value):
  a severity chosen once based on how disruptive that kind of issue typically is to
  downstream analysis.

Ambiguous cases (negative numbers, unusual-but-plausible dates, duplicate identifiers,
statistical outliers) are always capped at Low or Medium, with a "review, don't assume
it's wrong" recommendation.
"""

st.set_page_config(page_title="Data Quality Agent", page_icon="◆", layout="wide")
inject_global_css()


def _parse_uploaded_file(uploaded_file) -> LoadedDataset:
    """Parse a Streamlit UploadedFile into a LoadedDataset (sheet already resolved).

    Raises:
        FileLoadError: If the file cannot be parsed.
    """
    filename = uploaded_file.name
    is_excel = any(filename.lower().endswith(ext) for ext in EXCEL_EXTENSIONS)

    if not is_excel:
        return load_csv(uploaded_file, filename)

    sheet_names = get_excel_sheet_names(uploaded_file, filename)
    chosen_sheet = sheet_names[0]
    if len(sheet_names) > 1:
        chosen_sheet = st.sidebar.selectbox("This workbook has multiple sheets - select one", sheet_names)
    return load_excel(uploaded_file, filename, sheet_name=chosen_sheet)


def render_sidebar() -> tuple[str, LoadedDataset | None]:
    """Render the sidebar: branding, file upload (always mounted), and page navigation.

    The uploader lives in the sidebar (rather than a page's main content) so
    it stays mounted -- and the parsed dataset stays available in
    st.session_state -- no matter which workflow page is selected.

    Returns:
        (selected_page, loaded_dataset_or_None).
    """
    st.sidebar.markdown(
        '<div class="dqa-sidebar-title">Data Quality Agent</div>'
        '<div class="dqa-sidebar-subtitle">Human-guided data cleaning</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "txt", "xlsx", "xls"])

    result: LoadedDataset | None = None
    if uploaded_file is not None:
        # A real upload always wins over the demo dataset.
        st.session_state["use_demo_dataset"] = False
        st.sidebar.caption(human_readable_size(uploaded_file.size))
        try:
            result = _parse_uploaded_file(uploaded_file)
        except FileLoadError as exc:
            st.sidebar.error(str(exc))
        else:
            for warning in result.warnings:
                st.sidebar.warning(warning)
    elif st.session_state.get("use_demo_dataset"):
        # Loaded through the exact same load_csv() pipeline as any uploaded
        # file -- no separate demo workflow.
        try:
            result = load_csv(DEMO_DATASET_PATH, DEMO_DATASET_PATH.name)
        except FileLoadError as exc:
            st.sidebar.error(str(exc))
            st.session_state["use_demo_dataset"] = False
        else:
            for warning in result.warnings:
                st.sidebar.warning(warning)
            st.sidebar.success("Demo dataset loaded")

    # A new file, or a new sheet chosen for the same file, invalidates any
    # cleaning/session state that referred to the previous dataset.
    dataset_key = f"{result.file_name}:{result.dataframe.shape}:{result.sheet_name}" if result is not None else None
    if st.session_state.get("loaded_dataset_key") != dataset_key:
        st.session_state["loaded_dataset_key"] = dataset_key
        st.session_state.pop("cleaning_result", None)
        st.session_state.pop("cleaning_dataset_key", None)
    st.session_state["loaded_dataset"] = result

    # Jump to Upload's result page (Review Issues comes next, but Upload
    # itself now shows the overview once a file is loaded) -- exactly once,
    # the moment a file first loads, never again afterward, so it doesn't
    # fight the user's own navigation. The chosen page must be written back
    # to session_state immediately: any later rerun that isn't itself a nav
    # click (e.g. toggling a checkbox) would otherwise fall back to the
    # NAV_PAGES[0] default every time, since sidebar_nav only persists
    # nav_page when a button click changes it.
    had_file_before = st.session_state.get("_had_file", False)
    current_page = st.session_state.get("nav_page", NAV_PAGES[0])
    if result is not None and not had_file_before:
        current_page = "Upload"
        st.session_state["nav_page"] = current_page
    st.session_state["_had_file"] = result is not None

    st.sidebar.divider()
    page = sidebar_nav(PRIMARY_NAV_PAGES, SECONDARY_NAV_PAGES, current_page, _completed_primary_pages())

    return page, result


def _completed_primary_pages() -> set[str]:
    """Which of the four primary workflow steps have been completed this session.

    Each reflects an action actually taken (a file loaded, a page visited,
    a fix applied) -- never assumed -- so the sidebar's checkmarks track
    real progress. This is the app's only workflow-progress indicator
    (Phase 8.2 removed the separate stepper); the nav itself *is* the steps.
    """
    result = st.session_state.get("loaded_dataset")
    visited = st.session_state.get("visited_pages", set())
    completed = set()
    if result is not None:
        completed.add("Upload")
    if "Review Issues" in visited:
        completed.add("Review Issues")
    if st.session_state.get("cleaning_result") is not None:
        completed.add("Clean Data")
    if "Results" in visited:
        completed.add("Results")
    return completed


def render_dataset_overview(result: LoadedDataset) -> None:
    """Render row/column/missing/duplicate metrics and parsing metadata.

    No page_header of its own -- called from within the Upload page
    (Phase 8.2), which already rendered the page's title/subtitle.
    """
    df = result.dataframe

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    col4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

    details: dict[str, str] = {
        "File name": result.file_name,
        "File type": result.file_type,
    }
    if result.sheet_name:
        details["Sheet"] = result.sheet_name
    if result.encoding_used:
        details["Encoding used"] = result.encoding_used
    if result.delimiter_used:
        details["Delimiter used"] = repr(result.delimiter_used)

    with st.expander("File details"):
        st.json(details)


def render_preview(result: LoadedDataset) -> None:
    """Render a preview table of the first rows of the dataset."""
    st.markdown("#### Dataset Preview")
    expandable_table(result.dataframe, preview_rows=10, key="preview_view_all", width="stretch")


def _get_effective_profiles(df: pd.DataFrame) -> list[ColumnProfile]:
    """Profile a DataFrame and apply any type overrides recorded in this session.

    Overrides are stored in st.session_state["type_overrides"] as
    {column_name: selected_label} and never touch ``df`` itself -- this only
    changes each profile's ``effective_logical_type`` (see src/type_override.py).
    """
    profiles = profile_dataframe(df)
    overrides: dict[str, str] = st.session_state.get("type_overrides", {})
    if not overrides:
        return profiles
    return apply_type_overrides(profiles, overrides)


def render_review_issues_page(result: LoadedDataset) -> None:
    """Render the Review Issues page: what the agent found, and is its interpretation correct?

    Merges what used to be two separate pages -- Column Profiling and Data
    Quality Issues (Phase 8.2 consolidation). Section A answers "did the
    agent understand this column correctly?" (type confirmation/override);
    Section B answers "what quality problems exist, and how serious are
    they?" Both read the same effective-type-aware detection pass, so a
    type override made in Section A is immediately reflected in Section B.
    """
    page_header("Review Issues", "What the agent found, and whether its interpretation is correct.", badge=result.file_name)

    profiles = _get_effective_profiles(result.dataframe)
    detection = detect_issues(result.dataframe, profiles, result.file_name)
    issues = detection.issues
    summary = detection.summary

    high_critical_count = sum(1 for i in issues if i.severity in ("Critical", "High"))
    review_required_count = sum(1 for i in issues if not i.safe_to_auto_fix)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Issues", f"{summary.total_issues:,}")
    col2.metric("Columns Affected", len({i.column_name for i in issues if i.column_name}))
    col3.metric("High / Critical", f"{high_critical_count:,}")
    col4.metric("Needs Review", f"{review_required_count:,}")

    with st.popover("How is severity determined?"):
        st.markdown(_SEVERITY_EXPLANATION)

    if issues:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Issues by category**")
            st.dataframe(
                pd.DataFrame(sorted(summary.by_category.items()), columns=["Category", "Count"]),
                width='stretch', hide_index=True,
            )
        with col_b:
            st.markdown("**Issues by severity**")
            severity_rows = [(s, summary.by_severity[s]) for s in _SEVERITY_ORDER if s in summary.by_severity]
            severity_table = pd.DataFrame(severity_rows, columns=["Severity", "Count"])
            st.dataframe(style_severity_column(severity_table), width='stretch', hide_index=True)

    st.divider()
    st.markdown("### A. Review a Column")
    st.caption("Confirm the agent inferred the right type before it's used for cleaning.")

    type_counts = Counter(p.effective_logical_type for p in profiles)
    tcol, tpop = st.columns([0.7, 0.3])
    tcol.markdown(f"**{len(profiles)} columns** &nbsp;·&nbsp; **{len(type_counts)}** detected data types")
    with tpop.popover("View type breakdown"):
        for type_name, count in sorted(type_counts.items(), key=lambda kv: -kv[1]):
            st.markdown(f"- **{count}** {type_name}")

    column_names = [p.original_name for p in profiles]
    selected_name = st.selectbox("Select a column to inspect", column_names)
    selected_profile = next(p for p in profiles if p.original_name == selected_name)

    kpi_card_row(
        [
            {"label": "Detected Type", "value": selected_profile.logical_type, "accent": "primary"},
            {"label": "Effective Type", "value": selected_profile.effective_logical_type, "accent": "primary"},
            {"label": "Confidence", "value": f"{selected_profile.confidence:.0%}", "accent": "neutral"},
            {"label": "Missing %", "value": f"{selected_profile.missing_pct}%", "accent": "neutral"},
            {"label": "Unique Values", "value": f"{selected_profile.unique_count:,}", "accent": "neutral"},
        ]
    )

    render_type_override_controls(selected_profile)

    summary_df = pd.DataFrame(
        [
            {
                "Column": p.original_name,
                "Detected type": p.logical_type,
                "Confidence": p.confidence,
                "Effective type": p.effective_logical_type,
                "Overridden": p.type_overridden,
                "Pandas dtype": p.pandas_dtype,
                "Non-null": p.non_null_count,
                "Missing %": p.missing_pct,
                "Unique": p.unique_count,
                "Unique ratio": p.unique_ratio,
            }
            for p in profiles
        ]
    )
    with st.expander("View all column profiles"):
        st.dataframe(summary_df, width='stretch', hide_index=True)

    render_column_detail(selected_profile, result.dataframe[selected_name])

    st.divider()
    st.markdown("### B. Explore Detected Issues")

    if not issues:
        st.success("No quality issues detected. Your dataset passed the current quality checks.")
        return

    columns_with_issues = sorted({i.column_name for i in issues if i.column_name})
    categories = sorted({i.issue_category for i in issues})
    issue_types = sorted({i.issue_type for i in issues})
    severities = [s for s in _SEVERITY_ORDER if s in summary.by_severity]

    with st.container(border=True, key="card_issues_filters"):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        selected_columns = fcol1.multiselect("Column", columns_with_issues)
        selected_categories = fcol2.multiselect("Category", categories)
        selected_types = fcol3.multiselect("Issue type", issue_types, format_func=humanize_issue_type)
        selected_severities = fcol4.multiselect("Severity", severities)

    filtered = issues
    if selected_columns:
        filtered = [i for i in filtered if i.column_name in selected_columns]
    if selected_categories:
        filtered = [i for i in filtered if i.issue_category in selected_categories]
    if selected_types:
        filtered = [i for i in filtered if i.issue_type in selected_types]
    if selected_severities:
        filtered = [i for i in filtered if i.severity in selected_severities]

    st.caption(f"{len(filtered):,} of {summary.total_issues:,} issues match.")

    with st.expander(f"View detailed issues ({len(filtered):,})"):
        preview = _issues_to_table(filtered[:15])
        st.dataframe(style_severity_column(preview), width='stretch', hide_index=True)
        if len(filtered) > 15 and st.checkbox(f"View more rows ({len(filtered):,} total)", key="issues_table_view_all"):
            capped = filtered[:DEFAULT_MAX_ISSUES_DISPLAYED]
            if len(filtered) > DEFAULT_MAX_ISSUES_DISPLAYED:
                st.caption(f"Showing the first {DEFAULT_MAX_ISSUES_DISPLAYED:,} of {len(filtered):,} matching issues.")
            st.dataframe(style_severity_column(_issues_to_table(capped)), width='stretch', hide_index=True)

        st.markdown("**Understanding these issue types**")
        for issue_type in sorted({i.issue_type for i in filtered}):
            st.markdown(f"- **{humanize_issue_type(issue_type)}**: {ISSUE_TYPE_EXPLANATIONS.get(issue_type, 'No description available.')}")

    with st.expander("Inspect sample rows"):
        icol1, icol2 = st.columns(2)
        inspect_column = icol1.selectbox("Column", ["(any)"] + columns_with_issues, key="inspect_column")
        inspect_type = icol2.selectbox(
            "Issue type", ["(any)"] + issue_types, key="inspect_type",
            format_func=lambda t: t if t == "(any)" else humanize_issue_type(t),
        )

        row_level_candidates = [i for i in issues if i.row_index is not None]
        if inspect_column != "(any)":
            row_level_candidates = [i for i in row_level_candidates if i.column_name == inspect_column]
        if inspect_type != "(any)":
            row_level_candidates = [i for i in row_level_candidates if i.issue_type == inspect_type]

        affected_row_indices = sorted({i.row_index for i in row_level_candidates})
        if affected_row_indices:
            sample = affected_row_indices[:MAX_SAMPLE_ROWS_TO_INSPECT]
            st.caption(f"Showing {len(sample)} of {len(affected_row_indices):,} affected row(s).")
            st.dataframe(result.dataframe.loc[sample], width='stretch')
        else:
            st.info("No row-level issues match this selection.")


def render_type_override_controls(profile: ColumnProfile) -> None:
    """Render the human-in-the-loop type override UI for one column.

    Never modifies the raw dataset -- only st.session_state["type_overrides"]
    (which downstream issue detection reads via _get_effective_profiles) and
    st.session_state["type_override_log"] (an in-session audit trail).
    """
    with st.container(border=True, key="card_type_override"):
        dcol1, dcol2 = st.columns(2)
        dcol1.markdown(f"Detected as: **{profile.logical_type}**")
        dcol2.markdown(f"Confidence: **{profile.confidence:.0%}**")

        if is_low_confidence(profile.confidence, TYPE_OVERRIDE_CONFIDENCE_WARNING_THRESHOLD):
            st.warning("Low confidence -- please review this column before applying type-specific cleaning.")

        overrides: dict[str, str] = st.session_state.setdefault("type_overrides", {})
        current_label = overrides.get(profile.original_name, USE_DETECTED_LABEL)
        default_index = OVERRIDE_LABELS.index(current_label) if current_label in OVERRIDE_LABELS else 0

        selected_label = st.selectbox(
            "Confirm or change column type",
            OVERRIDE_LABELS,
            index=default_index,
            key=f"type_override_select_{profile.original_name}",
            help="Only affects which checks run in this session -- the uploaded data is never modified.",
        )

        if selected_label != current_label:
            record = make_override_record(profile, selected_label)
            st.session_state.setdefault("type_override_log", []).append(record)
            if selected_label == USE_DETECTED_LABEL:
                overrides.pop(profile.original_name, None)
            else:
                overrides[profile.original_name] = selected_label
            st.rerun()

        if profile.type_overridden:
            rcol1, rcol2 = st.columns([0.7, 0.3])
            rcol1.caption(f"Effective type for this session: **{profile.effective_logical_type}** (overridden).")
            if rcol2.button("Reset Type Override", key=f"reset_override_{profile.original_name}"):
                overrides.pop(profile.original_name, None)
                st.rerun()

    column_log = [r for r in st.session_state.get("type_override_log", []) if r.column_name == profile.original_name]
    if column_log:
        with st.expander("Type decision history for this column"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Timestamp": r.timestamp, "Original type": r.original_type,
                            "Original confidence": r.original_confidence, "New type": r.new_type,
                            "User approved": r.user_approved,
                        }
                        for r in column_log
                    ]
                ),
                width='stretch', hide_index=True,
            )


def render_column_detail(profile: ColumnProfile, series: pd.Series) -> None:
    """Render additional profile detail beyond the summary cards, evidence, and sample values."""
    with st.container(border=True, key="card_column_detail"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Original name:** {profile.original_name}")
            st.markdown(f"**Normalized name:** `{profile.normalized_name}`")
            st.markdown(f"**Pandas dtype:** `{profile.pandas_dtype}`")
        with col2:
            st.markdown(f"**Non-null values:** {profile.non_null_count:,}")
            st.markdown(f"**Duplicate values:** {profile.duplicate_value_count:,}")
            st.markdown(f"**Unique ratio:** {profile.unique_ratio}")

        if profile.min_value is not None:
            st.markdown(f"**Min:** {profile.min_value}  |  **Max:** {profile.max_value}")
            st.markdown(f"**Mean:** {profile.mean}  |  **Median:** {profile.median}  |  **Std dev:** {profile.std}")
            if profile.outlier_count is not None:
                st.markdown(f"**Possible outliers (IQR method):** {profile.outlier_count}")

        if profile.min_date is not None:
            st.markdown(f"**Earliest:** {profile.min_date}  |  **Latest:** {profile.max_date}")

        if profile.avg_text_length is not None:
            st.markdown(
                f"**Text length:** min {profile.min_text_length:.0f}, "
                f"max {profile.max_text_length:.0f}, avg {profile.avg_text_length}"
            )

        with st.expander("Technical detection details"):
            st.json(profile.evidence)

        st.markdown("**Sample values**")
        st.write(profile.example_values)

    value_counts = series.dropna().astype(str).value_counts().head(10)
    if not value_counts.empty:
        st.markdown("**Most frequent values**")
        st.dataframe(value_counts.rename("Count"), width='stretch')


def _issues_to_table(issues: list[Issue]) -> pd.DataFrame:
    """Convert a list of Issue records into a display-friendly DataFrame."""
    return pd.DataFrame(
        [
            {
                "Column": i.column_name,
                "Row": i.row_index,
                "Category": i.issue_category,
                "Issue type": humanize_issue_type(i.issue_type),
                "Severity": i.severity,
                "Confidence": i.confidence,
                "Current value": i.current_value,
                "Suggested value": i.suggested_value,
                "Safe to auto-fix": i.safe_to_auto_fix,
                "Recommended action": i.recommended_action,
            }
            for i in issues
        ]
    )


def _audit_log_to_df(entries: list[AuditLogEntry]) -> pd.DataFrame:
    """Convert audit log entries into a display/export-friendly DataFrame."""
    return pd.DataFrame(
        [
            {
                "Run ID": e.run_id,
                "Timestamp": e.timestamp,
                "Dataset": e.dataset_name,
                "Row": e.row_index,
                "Column": e.column_name,
                "Original value": e.original_value,
                "New value": e.new_value,
                "Cleaning action": e.cleaning_action,
                "Rule": e.rule_name,
                "Reason": e.reason,
                "User approved": e.user_approved,
                "Confidence": e.confidence,
            }
            for e in entries
        ]
    )


# Presentation-only unit for each safe fix's affected count -- purely a
# display choice (never shown as a bare floating number); the counts
# themselves still come straight from cleaning_engine.preview_fixes.
_FIX_AFFECTED_UNITS: dict[str, str] = {
    "trim_whitespace": "values",
    "collapse_internal_spaces": "values",
    "remove_non_printable": "values",
    "nullify_blank_and_placeholders": "values",
    "normalize_column_names": "columns",
    "remove_empty_rows": "rows",
    "remove_empty_columns": "columns",
    "convert_dates_stored_as_text": "values",
    "convert_numeric_stored_as_text": "values",
    "remove_exact_duplicate_rows": "rows",
}


def _format_affected_count(fix_id: str, count: int) -> str:
    """Render a safe fix's affected count with its unit, e.g. '12 columns' -- never a bare number."""
    unit = _FIX_AFFECTED_UNITS.get(fix_id, "values")
    return f"{count:,} {unit}"


def render_data_cleaning(result: LoadedDataset) -> CleaningResult | None:
    """Render the Clean Data page: safe automatic fixes, then guided issue review.

    Returns the active CleaningResult (if any fix or guided decision has
    changed data this session for the current dataset) -- the Results page
    uses it for the before/after dashboard and downloads.
    """
    page_header("Clean Data", "Review suggested fixes and decide how your data should be cleaned.", badge=result.file_name)
    st.caption("Works on a copy -- nothing changes until you approve it.")

    original_df = result.dataframe
    dataset_key = f"{result.file_name}:{original_df.shape}:{result.sheet_name}"
    if st.session_state.get("cleaning_dataset_key") != dataset_key:
        st.session_state["cleaning_dataset_key"] = dataset_key
        st.session_state.pop("cleaning_result", None)
        st.session_state.pop("cleaning_decision_log", None)

    st.markdown("#### A. Safe automatic fixes")
    previews = preview_fixes(original_df, result.file_name)
    active_previews = [p for p in previews if p.affected_count > 0]
    zero_impact_previews = [p for p in previews if p.affected_count == 0]

    selected_fix_ids: list[str] = []
    if not active_previews:
        st.caption("No safe automatic fixes apply to this dataset right now.")
    for preview in active_previews:
        with st.container(border=True, key=f"card_fix_{preview.fix_id}"):
            row = st.columns([0.06, 0.94])
            checked = row[0].checkbox("Select", key=f"fix_{preview.fix_id}", label_visibility="collapsed")
            label = preview.title
            if preview.requires_explicit_approval:
                label += " ⚠️"
            count_label = _format_affected_count(preview.fix_id, preview.affected_count)
            row[1].markdown(f"**{label}** &nbsp;&nbsp;·&nbsp;&nbsp; {count_label}")
            row[1].caption(preview.description)
            if preview.notes:
                row[1].warning(" ".join(preview.notes))
            if checked:
                selected_fix_ids.append(preview.fix_id)
    st.caption("Counts assume fixes apply in the order shown.")

    if zero_impact_previews:
        with st.expander(f"Other available checks ({len(zero_impact_previews)})"):
            for preview in zero_impact_previews:
                zero_label = _format_affected_count(preview.fix_id, 0)
                st.caption(f"**{preview.title}** -- {preview.description} ({zero_label})")

    apply_col, reset_col = st.columns(2)
    if apply_col.button("Apply Selected Fixes", type="primary", disabled=not selected_fix_ids):
        existing = st.session_state.get("cleaning_result")
        base_df = existing.cleaned_df if existing is not None else original_df
        fix_result = apply_selected_fixes(base_df, selected_fix_ids, result.file_name)
        st.session_state["cleaning_result"] = _merge_cleaning_results(existing, fix_result)
        st.rerun()
    if reset_col.button("Reset to Original Uploaded Dataset"):
        st.session_state.pop("cleaning_result", None)
        st.session_state.pop("cleaning_decision_log", None)
        st.rerun()

    cleaning_result: CleaningResult | None = st.session_state.get("cleaning_result")
    if cleaning_result is not None:
        st.success(f"Working copy has {len(cleaning_result.audit_log):,} logged change(s) so far this session.")
        for fix_id, notes in cleaning_result.skipped_ineligible:
            for note in notes:
                st.warning(note)

    current_df = cleaning_result.cleaned_df if cleaning_result is not None else original_df

    st.divider()
    st.markdown("#### B. Guided issue review")
    st.caption("Review one issue at a time. Nothing changes until you apply it.")
    render_guided_issue_review(result, current_df)

    cleaning_result = st.session_state.get("cleaning_result")
    if cleaning_result is None:
        st.info("No fixes or cleaning decisions applied yet.")
        return None

    st.divider()
    render_before_after_comparison(original_df, cleaning_result.cleaned_df)
    render_changed_rows_preview(original_df, cleaning_result)

    st.markdown("#### Cleaning logs")
    audit_count = len(cleaning_result.audit_log)
    decision_count = len(st.session_state.get("cleaning_decision_log", []))
    st.caption(f"{audit_count:,} value(s) changed · {decision_count:,} human decision(s) recorded")
    render_audit_log(cleaning_result)
    render_cleaning_decision_log()

    st.divider()
    st.info("See the full before/after breakdown and download your cleaned file on the **Results** page.")

    return cleaning_result


def _merge_cleaning_results(existing: CleaningResult | None, new_result: CleaningResult) -> CleaningResult:
    """Fold a freshly-applied fix/decision result into the running session CleaningResult.

    Keeps the *first* run_id/timestamp for the session (so every fix and
    guided decision in one working session shares one run_id, matching
    Phase 5's "one run = one Save Run to Database entry" model) while
    accumulating audit log entries and applied fix ids across calls.
    """
    if existing is None:
        return new_result
    return dataclasses_replace(
        existing,
        cleaned_df=new_result.cleaned_df,
        audit_log=existing.audit_log + new_result.audit_log,
        applied_fix_ids=existing.applied_fix_ids + [f for f in new_result.applied_fix_ids if f not in existing.applied_fix_ids],
        skipped_ineligible=new_result.skipped_ineligible,
    )


def _get_or_create_cleaning_result(result: LoadedDataset) -> CleaningResult:
    """Return the session's current CleaningResult, or bootstrap an empty one.

    Bootstrapping lets a guided decision be applied even if no safe fix has
    been applied yet -- the working copy starts as an exact copy of the
    original with an empty audit log.
    """
    existing = st.session_state.get("cleaning_result")
    if existing is not None:
        return existing
    return CleaningResult(
        run_id=str(uuid.uuid4()), dataset_name=result.file_name, timestamp=datetime.now(timezone.utc).isoformat(),
        cleaned_df=result.dataframe.copy(), audit_log=[], applied_fix_ids=[], skipped_ineligible=[],
    )


def render_guided_issue_review(result: LoadedDataset, current_df: pd.DataFrame) -> None:
    """Render the guided, human-in-the-loop issue review workflow (Phase 7B).

    Re-profiles and re-detects issues on the *current* working copy every
    render, so a group's affected count and effective type always reflect
    whatever has already been applied this session -- then groups them by
    (column, issue_type) and lets the user work through one group at a time.
    """
    profiles = _get_effective_profiles(current_df)
    profiles_by_name = {p.original_name: p for p in profiles}
    issues = detect_issues(current_df, profiles, result.file_name).issues
    groups = build_issue_groups(issues, profiles_by_name)

    if not groups:
        st.success("No guided-review issues remain (missing values, outliers, negative values, or category variants).")
        return

    labels = [f"{g.column_name} -- {humanize_issue_type(g.issue_type)} ({g.affected_count} affected, {g.severity})" for g in groups]
    selected_idx = st.selectbox(
        "Choose an issue group to review", range(len(groups)), format_func=lambda i: labels[i], key="guided_group_select",
    )
    with st.container(border=True, key="card_guided_review"):
        render_issue_group_card(result, current_df, groups[selected_idx])


def render_issue_group_card(result: LoadedDataset, current_df: pd.DataFrame, group: IssueGroup) -> None:
    """Render one issue group: explanation, affected sample, strategy choice, preview, apply."""
    st.markdown(f"##### {group.column_name}")
    st.markdown(
        f"{group.effective_logical_type} · {group.inference_confidence:.0%} confidence &nbsp;&nbsp; "
        f"{humanize_issue_type(group.issue_type)} · {group.affected_count:,} affected &nbsp;&nbsp; "
        + severity_badge(group.severity),
        unsafe_allow_html=True,
    )

    sample_indices = group.row_indices[:QUICK_PREVIEW_ROWS]
    if sample_indices:
        with st.expander(f"Sample affected rows ({len(sample_indices)} of {group.affected_count:,})"):
            st.dataframe(current_df.loc[sample_indices], width='stretch')

    effective_type = LogicalType(group.effective_logical_type)
    st.markdown("**Choose cleaning strategy**")

    if group.issue_type == "missing_null":
        render_missing_value_review(result, current_df, group, effective_type)
    elif group.issue_type == "possible_outlier":
        render_outlier_review(result, current_df, group, effective_type)
    elif group.issue_type == "negative_value":
        render_negative_value_review(result, current_df, group, effective_type)
    elif group.issue_type in ("similar_category_values", "inconsistent_capitalization"):
        render_categorical_variant_review(result, current_df, group, effective_type)
    else:
        st.info("No guided strategy is defined for this issue type yet.")


def _format_stat_value(value: Any) -> str:
    """Format one statistic for a compact metric card."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _render_stat_cards(stats: dict[str, Any]) -> None:
    """Render a row of compact metric cards for whichever stats are present (non-None)."""
    present = [(key, value) for key, value in stats.items() if value is not None]
    if not present:
        return
    cols = st.columns(len(present))
    for col, (key, value) in zip(cols, present):
        col.metric(key.replace("_", " ").title(), _format_stat_value(value))


def render_missing_value_review(
    result: LoadedDataset, current_df: pd.DataFrame, group: IssueGroup, effective_type: LogicalType
) -> None:
    """Missing-value guided review: strategy choice per src.cleaning_strategies, then preview/apply."""
    series = current_df[group.column_name]
    catalog = get_missing_value_strategies(effective_type, series)

    if not catalog.options:
        st.warning(catalog.note or "No strategies available for this column's type.")
        return

    if catalog.stats:
        st.caption("Relevant statistics")
        _render_stat_cards(catalog.stats)

    key_prefix = f"missing_{group.column_name}"
    strategy = _render_strategy_selector(catalog.options, key_prefix)

    scope, selected_rows = "n/a", []
    per_row_values: dict[int, str] = {}
    custom_value: Optional[str] = None

    if strategy.strategy_id not in ("keep_as_null", "do_not_clean"):
        scope, selected_rows = _render_scope_selector(group, key_prefix)
        if strategy.requires_row_scope:
            per_row_values = _render_per_row_value_editor(current_df, group.column_name, selected_rows, key_prefix)
        elif strategy.requires_custom_value:
            custom_value = _render_custom_value_input(strategy, key_prefix)

    replacements = _resolve_missing_value_replacements(
        strategy, selected_rows, catalog.stats, per_row_values, custom_value, effective_type
    )
    _render_preview_and_apply(
        result, current_df, group, strategy=strategy, scope=scope, replacements=replacements,
        key_prefix=key_prefix, custom_value=custom_value,
    )


def render_outlier_review(
    result: LoadedDataset, current_df: pd.DataFrame, group: IssueGroup, effective_type: LogicalType
) -> None:
    """Outlier guided review: never implies an outlier must be changed."""
    st.warning("A statistical outlier is not necessarily an error. Review the values before deciding whether to modify them.")

    series = current_df[group.column_name]
    catalog = get_outlier_strategies(effective_type, series)
    if not catalog.options:
        st.info(catalog.note or "Outlier review is not available for this column's type.")
        return

    stats = catalog.stats
    st.caption("Relevant statistics")
    _render_stat_cards(stats)

    key_prefix = f"outlier_{group.column_name}"
    strategy = _render_strategy_selector(catalog.options, key_prefix)

    scope, selected_rows = "n/a", []
    custom_value: Optional[str] = None
    replacements: dict[int, Optional[object]] = {}

    if strategy.strategy_id not in ("keep_outlier", "do_not_clean"):
        scope, selected_rows = _render_scope_selector(group, key_prefix)
        if strategy.requires_custom_value:
            custom_value = _render_custom_value_input(strategy, key_prefix)
        numeric_series = numeric_values(series, effective_type)
        replacements = _resolve_outlier_replacements(strategy, selected_rows, numeric_series, stats, custom_value)

    _render_preview_and_apply(
        result, current_df, group, strategy=strategy, scope=scope, replacements=replacements,
        key_prefix=key_prefix, custom_value=custom_value,
    )


def render_negative_value_review(
    result: LoadedDataset, current_df: pd.DataFrame, group: IssueGroup, effective_type: LogicalType
) -> None:
    """Two-stage negative-value review: valid-or-invalid decision, then optional treatment."""
    st.info("Negative values detected. Negative values may be valid for this column.")

    key_prefix = f"negative_{group.column_name}"
    decision = _render_strategy_selector(get_negative_value_decision_options(), key_prefix)

    if decision.strategy_id == "negative_valid":
        _render_preview_and_apply(
            result, current_df, group, strategy=decision, scope="all", replacements={}, key_prefix=key_prefix,
        )
        return

    series = current_df[group.column_name]
    catalog = get_negative_value_treatment_strategies(effective_type, series)
    if not catalog.options:
        st.info("No numeric treatment strategies available for this column's type.")
        return

    stats = catalog.stats
    if stats:
        st.caption("Relevant statistics (of current values)")
        _render_stat_cards(stats)

    treat_prefix = f"{key_prefix}_treat"
    treatment = _render_strategy_selector(catalog.options, treat_prefix)
    scope, selected_rows = _render_scope_selector(group, treat_prefix)
    custom_value = _render_custom_value_input(treatment, treat_prefix) if treatment.requires_custom_value else None

    replacements = _resolve_negative_treatment_replacements(treatment, selected_rows, stats, custom_value)
    _render_preview_and_apply(
        result, current_df, group, strategy=treatment, scope=scope, replacements=replacements,
        key_prefix=treat_prefix, custom_value=custom_value,
    )


def render_categorical_variant_review(
    result: LoadedDataset, current_df: pd.DataFrame, group: IssueGroup, effective_type: LogicalType
) -> None:
    """Category-variant guided review: never auto-merges, always shows variants before applying."""
    series = current_df[group.column_name]
    variant_rows = group.row_indices
    variants = sorted({str(series.loc[r]) for r in variant_rows if r in series.index and pd.notna(series.loc[r])})
    st.caption(f"Detected variants: {', '.join(variants)}" if variants else "No variant values available to display.")

    key_prefix = f"variant_{group.column_name}"
    strategy = _render_strategy_selector(get_categorical_variant_strategies(), key_prefix)

    replacements: dict[int, Optional[object]] = {}
    custom_value: Optional[str] = None

    if strategy.strategy_id == "standardize_most_frequent":
        # Reuse the canonical value issue_detector already computed per row's
        # similarity cluster (Phase 3) -- a group can contain more than one
        # cluster, so each issue's own suggested_value is used, not one
        # single value for the whole group.
        replacements = {
            i.row_index: i.suggested_value
            for i in group.issues
            if i.row_index is not None and i.suggested_value and str(series.get(i.row_index)) != i.suggested_value
        }
    elif strategy.strategy_id == "select_canonical":
        canonical = st.selectbox("Canonical value", variants, key=f"{key_prefix}_canonical") if variants else None
        custom_value = canonical
        if canonical:
            replacements = {r: canonical for r in variant_rows if r in series.index and str(series.loc[r]) != canonical}
    elif strategy.strategy_id == "custom_canonical":
        custom_value = st.text_input("Custom canonical value", key=f"{key_prefix}_custom_value")
        if custom_value:
            replacements = {r: custom_value for r in variant_rows}

    scope = "all" if replacements else "n/a"
    _render_preview_and_apply(
        result, current_df, group, strategy=strategy, scope=scope, replacements=replacements,
        key_prefix=key_prefix, custom_value=custom_value,
    )


# --- Guided review shared helpers --------------------------------------------------------


def _render_strategy_selector(options: list[CleaningStrategyOption], key_prefix: str) -> CleaningStrategyOption:
    """Render a strategy dropdown in its own decision card; selecting an option never modifies data by itself."""
    with st.container(border=True, key=f"card_strategy_{key_prefix}"):
        labels = [o.title for o in options]
        selected_label = st.selectbox("Cleaning strategy", labels, key=f"{key_prefix}_strategy")
        st.caption("Choose how to handle this issue.")
    return next(o for o in options if o.title == selected_label)


def _render_scope_selector(group: IssueGroup, key_prefix: str) -> tuple[str, list[int]]:
    """Let the user apply a strategy to all affected rows or a hand-picked subset.

    Row-level selection is limited to the rows already shown in the sample
    above (capped at MAX_SAMPLE_ROWS_TO_INSPECT) -- never a picker over
    thousands of rows at once.
    """
    sample_indices = group.row_indices[:MAX_SAMPLE_ROWS_TO_INSPECT]
    scope_choice = st.radio(
        "Apply to", ["All affected rows", "Only selected affected rows"], key=f"{key_prefix}_scope", horizontal=True,
    )
    if scope_choice == "All affected rows":
        return "all", group.row_indices
    selected = st.multiselect(
        "Select affected row(s) (limited to the sample shown above)", sample_indices, key=f"{key_prefix}_selected_rows",
    )
    return "selected", selected


def _render_custom_value_input(strategy: CleaningStrategyOption, key_prefix: str) -> Optional[str]:
    """Render a single custom-value input appropriate to the strategy, as a string."""
    if strategy.strategy_id == "custom_numeric":
        value = st.number_input("Custom numeric value", key=f"{key_prefix}_custom_value", value=0.0, step=1.0)
        return str(value)
    if strategy.strategy_id == "custom_date":
        value = st.date_input("Custom date", key=f"{key_prefix}_custom_value")
        return value.isoformat() if isinstance(value, date) else str(value)
    value = st.text_input("Custom value", key=f"{key_prefix}_custom_value")
    return value if value.strip() else None


def _render_per_row_value_editor(
    current_df: pd.DataFrame, column: str, row_indices: list[int], key_prefix: str
) -> dict[int, str]:
    """Editable table for one custom value per row -- never a single bulk value for many rows.

    Capped at MAX_SAMPLE_ROWS_TO_INSPECT rows at a time so the editor never
    displays thousands of rows at once; re-run the review for further rows
    after applying (the group's affected rows shrink as each batch is applied).
    """
    if not row_indices:
        return {}
    capped = row_indices[:MAX_SAMPLE_ROWS_TO_INSPECT]
    if len(row_indices) > len(capped):
        st.caption(f"Showing the first {len(capped)} of {len(row_indices):,} row(s) -- apply in multiple passes for the rest.")
    edit_df = pd.DataFrame(
        {
            "Row": capped,
            "Current value": [current_df.loc[r, column] if r in current_df.index else None for r in capped],
            "New value": ["" for _ in capped],
        }
    )
    edited = st.data_editor(edit_df, hide_index=True, disabled=["Row", "Current value"], width='stretch', key=f"{key_prefix}_row_editor")
    return {int(row): value for row, value in zip(edited["Row"], edited["New value"]) if str(value).strip() != ""}


def _resolve_missing_value_replacements(
    strategy: CleaningStrategyOption,
    row_indices: list[int],
    stats: dict[str, Any],
    per_row_values: dict[int, str],
    custom_value: Optional[str],
    effective_type: LogicalType,
) -> dict[int, Optional[object]]:
    """Turn a chosen missing-value strategy into {row_index: new_value}."""
    if strategy.strategy_id in ("keep_as_null", "do_not_clean"):
        return {}
    if strategy.strategy_id == "custom_per_row":
        return {row: value for row, value in per_row_values.items() if row in row_indices and str(value).strip() != ""}

    fill_value: Optional[object]
    if strategy.strategy_id == "replace_mean":
        fill_value = stats.get("mean")
    elif strategy.strategy_id == "replace_median":
        fill_value = stats.get("median")
    elif strategy.strategy_id == "replace_mode":
        fill_value = stats.get("mode")
    elif strategy.strategy_id == "replace_zero":
        fill_value = 0
    elif strategy.strategy_id == "replace_unknown_label":
        fill_value = "Unknown"
    elif strategy.strategy_id == "replace_not_provided":
        fill_value = "Not Provided"
    elif strategy.strategy_id == "replace_true":
        fill_value = True
    elif strategy.strategy_id == "replace_false":
        fill_value = False
    elif strategy.strategy_id == "custom_numeric":
        fill_value = _parse_float_or_none(custom_value)
    elif strategy.strategy_id == "custom_text":
        fill_value = custom_value if custom_value not in (None, "") else None
    elif strategy.strategy_id == "custom_date":
        fill_value = _parse_custom_date(custom_value, effective_type)
    else:
        fill_value = None

    if fill_value is None:
        return {}
    return {row: fill_value for row in row_indices}


def _resolve_outlier_replacements(
    strategy: CleaningStrategyOption,
    row_indices: list[int],
    numeric_series: pd.Series,
    stats: dict[str, Any],
    custom_value: Optional[str],
) -> dict[int, Optional[object]]:
    """Turn a chosen outlier strategy into {row_index: new_value}."""
    if strategy.strategy_id == "set_null":
        return {row: None for row in row_indices}
    if strategy.strategy_id == "replace_median":
        median = stats.get("median")
        return {row: median for row in row_indices} if median is not None else {}
    if strategy.strategy_id == "iqr_cap":
        lower, upper = stats.get("lower_bound"), stats.get("upper_bound")
        if lower is None or upper is None:
            return {}
        replacements: dict[int, Optional[object]] = {}
        for row in row_indices:
            value = numeric_series.get(row)
            if value is None or pd.isna(value):
                continue
            if value < lower:
                replacements[row] = lower
            elif value > upper:
                replacements[row] = upper
        return replacements
    if strategy.strategy_id == "custom_numeric":
        value = _parse_float_or_none(custom_value)
        return {row: value for row in row_indices} if value is not None else {}
    return {}


def _resolve_negative_treatment_replacements(
    strategy: CleaningStrategyOption, row_indices: list[int], stats: dict[str, Any], custom_value: Optional[str]
) -> dict[int, Optional[object]]:
    """Turn a chosen negative-value treatment strategy into {row_index: new_value}."""
    if strategy.strategy_id == "set_null":
        return {row: None for row in row_indices}
    if strategy.strategy_id == "replace_zero":
        return {row: 0 for row in row_indices}
    if strategy.strategy_id == "replace_median":
        median = stats.get("median")
        return {row: median for row in row_indices} if median is not None else {}
    if strategy.strategy_id == "custom_numeric":
        value = _parse_float_or_none(custom_value)
        return {row: value for row in row_indices} if value is not None else {}
    return {}


def _parse_float_or_none(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _parse_custom_date(value: Optional[str], effective_type: LogicalType) -> Optional[object]:
    """Parse a user-entered date, matching the column's actual storage (real datetime vs. text)."""
    if value in (None, ""):
        return None
    if effective_type in (LogicalType.DATE, LogicalType.DATETIME):
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
    return value


def _render_preview_and_apply(
    result: LoadedDataset,
    current_df: pd.DataFrame,
    group: IssueGroup,
    *,
    strategy: CleaningStrategyOption,
    scope: str,
    replacements: dict[int, Optional[object]],
    key_prefix: str,
    custom_value: Optional[str] = None,
) -> None:
    """Preview the exact effect of a chosen strategy; apply only on explicit confirmation.

    Selecting a strategy or scope never modifies the working dataset --
    only clicking "Apply Cleaning Decision" does.
    """
    st.markdown("**Preview**")
    button_type = "primary"
    if not replacements:
        st.info(f"'{strategy.title}' makes no change to the data -- it will still be recorded as a decision.")
        button_type = "secondary"
    else:
        preview_rows = list(replacements.items())[:QUICK_PREVIEW_ROWS]
        preview_df = pd.DataFrame(
            [
                {
                    "Row": row,
                    "Current value": current_df.loc[row, group.column_name] if row in current_df.index else None,
                    "": "→",
                    "Proposed value": new_value,
                }
                for row, new_value in preview_rows
            ]
        )
        st.caption(f"{len(replacements):,} value(s) across {len(replacements):,} row(s) will change.")
        st.dataframe(preview_df, width='stretch', hide_index=True)
        if len(replacements) > len(preview_rows):
            st.caption(f"Showing {len(preview_rows)} of {len(replacements):,} changes in this preview.")
        if strategy.strategy_id == "set_null":
            st.warning(f"This permanently replaces {len(replacements):,} value(s) with a missing value (NULL).")

    if st.button("Apply Cleaning Decision", type=button_type, key=f"{key_prefix}_apply"):
        decision_result = strategy.title if not replacements else f"{strategy.title} ({len(replacements)} value(s) changed)"
        _apply_and_record_decision(
            result, group, strategy=strategy, scope=scope, replacements=replacements,
            decision_result=decision_result, custom_value=custom_value,
        )


def _apply_and_record_decision(
    result: LoadedDataset,
    group: IssueGroup,
    *,
    strategy: CleaningStrategyOption,
    scope: str,
    replacements: dict[int, Optional[object]],
    decision_result: str,
    custom_value: Optional[str] = None,
) -> None:
    """Log a confirmed guided-cleaning decision and, if it changes data, apply it.

    Every decision is recorded in the Cleaning Decision Log, including ones
    where ``replacements`` is empty ("Keep as NULL", "Do not clean", "Keep
    outlier(s)", "Negative values are valid", "Keep all variants") -- those
    are logged decisions, not applied fixes, and never touch the data.
    """
    cleaning_result = _get_or_create_cleaning_result(result)
    timestamp = datetime.now(timezone.utc).isoformat()

    new_df = cleaning_result.cleaned_df
    audit_entries: list[AuditLogEntry] = []
    if replacements:
        new_df, audit_entries = apply_value_replacements(
            cleaning_result.cleaned_df, group.column_name, replacements,
            result.file_name, cleaning_result.run_id, timestamp,
            cleaning_action=f"guided_{group.issue_type}_{strategy.strategy_id}",
            reason=strategy.description, confidence=group.inference_confidence,
        )

    st.session_state["cleaning_result"] = dataclasses_replace(
        cleaning_result, cleaned_df=new_df, audit_log=cleaning_result.audit_log + audit_entries,
    )

    decision_log: list[CleaningDecisionLogEntry] = st.session_state.get("cleaning_decision_log", [])
    st.session_state["cleaning_decision_log"] = decision_log + [
        CleaningDecisionLogEntry(
            run_id=cleaning_result.run_id, timestamp=timestamp, dataset_name=result.file_name,
            column_name=group.column_name, issue_type=group.issue_type,
            effective_logical_type=group.effective_logical_type, inference_confidence=group.inference_confidence,
            selected_strategy=strategy.title, scope=scope,
            affected_count=len(replacements) if replacements else group.affected_count,
            user_approved=True, decision_result=decision_result, custom_value=custom_value,
            reason=strategy.description,
        )
    ]
    st.rerun()


def render_cleaning_decision_log() -> None:
    """Render the separate log of human cleaning decisions (Phase 7B).

    Distinct from the audit log: this captures every reviewed decision,
    including ones that changed nothing, so an accepted NULL/outlier/
    negative value/variant is never mistaken for an unresolved issue.
    """
    decisions: list[CleaningDecisionLogEntry] = st.session_state.get("cleaning_decision_log", [])
    with st.expander(f"Decision Log — Human Decisions ({len(decisions):,})"):
        if not decisions:
            st.caption("No guided cleaning decisions recorded yet.")
        else:
            expandable_table(
                _decision_log_to_df(decisions), preview_rows=QUICK_PREVIEW_ROWS,
                key="decision_log_view_all", max_full_rows=DEFAULT_MAX_ISSUES_DISPLAYED, width='stretch',
            )


def _decision_log_to_df(entries: list[CleaningDecisionLogEntry]) -> pd.DataFrame:
    """Convert cleaning decision log entries into a display/export-friendly DataFrame."""
    return pd.DataFrame(
        [
            {
                "Run ID": e.run_id, "Timestamp": e.timestamp, "Dataset": e.dataset_name,
                "Column": e.column_name, "Issue type": e.issue_type,
                "Effective type": e.effective_logical_type, "Confidence": e.inference_confidence,
                "Strategy": e.selected_strategy, "Scope": e.scope, "Affected count": e.affected_count,
                "Approved": e.user_approved, "Decision result": e.decision_result,
                "Custom value": e.custom_value, "Reason": e.reason,
            }
            for e in entries
        ]
    )


def _paired_metric(col, label: str, before: int, after: int, *, fewer_is_better: Optional[bool] = None) -> None:
    """Render one 'before -> after' KPI card with the correct improvement color.

    A decrease is only shown as an improvement (green) when ``fewer_is_better``
    says so (e.g. missing values, duplicate rows) -- a shrinking row/column
    count is structural, not inherently good or bad, so it stays neutral gray.
    """
    delta = after - before
    delta_color = "off" if fewer_is_better is None or delta == 0 else ("inverse" if fewer_is_better else "normal")
    col.metric(label, f"{before:,} → {after:,}", delta=delta, delta_color=delta_color)
    if fewer_is_better and delta < 0:
        col.caption(f"{-delta:,} resolved")


def render_before_after_comparison(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
    """Render row/column/missing/duplicate counts and dtypes, original vs cleaned."""
    st.markdown("#### Before / after comparison")
    col1, col2, col3, col4 = st.columns(4)
    _paired_metric(col1, "Rows", original_df.shape[0], cleaned_df.shape[0])
    _paired_metric(col2, "Columns", original_df.shape[1], cleaned_df.shape[1])
    _paired_metric(col3, "Missing Values", int(original_df.isna().sum().sum()), int(cleaned_df.isna().sum().sum()), fewer_is_better=True)
    _paired_metric(col4, "Duplicate Rows", int(original_df.duplicated().sum()), int(cleaned_df.duplicated().sum()), fewer_is_better=True)

    with st.expander("Data types: original vs cleaned"):
        dcol1, dcol2 = st.columns(2)
        dcol1.markdown("**Original**")
        dcol1.dataframe(original_df.dtypes.astype(str).rename("dtype"), width='stretch')
        dcol2.markdown("**Cleaned**")
        dcol2.dataframe(cleaned_df.dtypes.astype(str).rename("dtype"), width='stretch')


def render_changed_rows_preview(original_df: pd.DataFrame, cleaning_result: CleaningResult) -> None:
    """Render a before/after preview of a sample of rows that were actually changed."""
    changed_indices = sorted({e.row_index for e in cleaning_result.audit_log if e.row_index is not None})
    with st.expander(f"View changed rows ({len(changed_indices):,})"):
        if not changed_indices:
            st.caption("No row-level changes (only column- or dataset-level fixes, if any).")
            return

        sample = changed_indices[:QUICK_PREVIEW_ROWS]
        st.caption(f"Showing {len(sample)} of {len(changed_indices):,} changed row(s).")

        cleaned_df = cleaning_result.cleaned_df
        still_present = [i for i in sample if i in cleaned_df.index]
        removed = [i for i in sample if i not in cleaned_df.index]

        if still_present:
            st.markdown("**Before**")
            st.dataframe(original_df.loc[still_present], width='stretch')
            st.markdown("**After**")
            st.dataframe(cleaned_df.loc[still_present], width='stretch')
        if removed:
            st.markdown(f"**Removed rows** ({len(removed)} shown)")
            st.dataframe(original_df.loc[removed], width='stretch')


def render_audit_log(cleaning_result: CleaningResult) -> None:
    """Render a sampled, filterable view of the full audit log."""
    with st.expander(f"Audit Log — Actual Value Changes ({len(cleaning_result.audit_log):,})"):
        if not cleaning_result.audit_log:
            st.caption("No changes were logged.")
            return
        st.caption(f"Run `{cleaning_result.run_id}`.")
        expandable_table(
            _audit_log_to_df(cleaning_result.audit_log), preview_rows=QUICK_PREVIEW_ROWS,
            key="audit_log_view_all", max_full_rows=DEFAULT_MAX_ISSUES_DISPLAYED, width='stretch',
        )


def render_primary_download(result: LoadedDataset, cleaning_result: CleaningResult) -> None:
    """Render the app's primary output: the cleaned CSV/Excel downloads, prominently.

    The cleaned dataset is the primary product of this whole application,
    so this section sits near the top of the Results page (right after the
    Cleaning Summary KPIs) with two full-width, primary-styled buttons --
    everything else (audit log, decision log, unresolved issues, applied
    fixes) is a secondary, visually quieter download (see
    render_secondary_downloads), further down the page.
    """
    with st.container(border=True, key="card_primary_download"):
        st.markdown("#### Your cleaned dataset is ready")
        st.caption("Download the reviewed and cleaned version of your data.")
        cleaned_df = cleaning_result.cleaned_df

        col1, col2 = st.columns(2)
        if col1.download_button(
            "Download Cleaned CSV", cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_dataset.csv", mime="text/csv", type="primary", icon="⬇️", width="stretch",
        ):
            st.session_state["exported_cleaned_file"] = True

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            cleaned_df.to_excel(writer, index=False, sheet_name="Cleaned")
        if col2.download_button(
            "Download Cleaned Excel", excel_buffer.getvalue(), file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary", icon="⬇️", width="stretch",
        ):
            st.session_state["exported_cleaned_file"] = True


def render_secondary_downloads(result: LoadedDataset, cleaning_result: CleaningResult) -> None:
    """Render the secondary, technical downloads: audit log, decision log, unresolved issues, fix summary.

    Deliberately lower on the Results page and visually quieter (tertiary
    buttons, tucked inside one expander) than render_primary_download --
    these are supporting evidence, not the product itself.
    """
    with st.expander("Additional downloads"):
        cleaned_df = cleaning_result.cleaned_df
        col3, col4, col5 = st.columns(3)

        audit_df = _audit_log_to_df(cleaning_result.audit_log)
        col3.download_button(
            "Audit Log", audit_df.to_csv(index=False).encode("utf-8"),
            file_name="audit_log.csv", mime="text/csv", type="tertiary",
        )

        decision_log = st.session_state.get("cleaning_decision_log", [])
        decision_df = _decision_log_to_df(decision_log) if decision_log else pd.DataFrame()
        col4.download_button(
            "Decision Log", decision_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaning_decision_log.csv", mime="text/csv", type="tertiary",
        )

        remaining = detect_issues(cleaned_df, profile_dataframe(cleaned_df), result.file_name)
        unresolved_df = _issues_to_table(remaining.issues)
        col5.download_button(
            "Unresolved Issues Report", unresolved_df.to_csv(index=False).encode("utf-8"),
            file_name="unresolved_issues.csv", mime="text/csv", type="tertiary",
        )

        st.markdown("**Applied-fixes summary**")
        fix_summary_counts = Counter(e.cleaning_action for e in cleaning_result.audit_log)
        fix_summary_df = pd.DataFrame(sorted(fix_summary_counts.items()), columns=["Cleaning action", "Number of changes"])
        st.dataframe(fix_summary_df, width='stretch', hide_index=True)
        st.download_button(
            "Applied-Fixes Summary", fix_summary_df.to_csv(index=False).encode("utf-8"),
            file_name="applied_fixes_summary.csv", mime="text/csv", type="tertiary",
        )


def render_run_history_page() -> None:
    """Render the Run History page: saved-run table, filters, comparison, and detail."""
    page_header("Run History", "Every saved run, loaded from SQLite -- survives closing and reopening the app.")
    render_run_history()


def render_component_comparison(original_score: QualityScore, cleaned_score: QualityScore | None) -> None:
    """Render a before/after table of the five component scores plus their explanations."""
    rows = [{"Component": c.component_name, "Weight %": c.weight, "Original score": c.score} for c in original_score.components]
    table = pd.DataFrame(rows).set_index("Component")

    if cleaned_score is not None:
        cleaned_by_name = {c.component_name: c.score for c in cleaned_score.components}
        table["Cleaned score"] = [cleaned_by_name[name] for name in table.index]
        table["Improvement"] = (table["Cleaned score"] - table["Original score"]).round(2)

    st.dataframe(table, width='stretch')

    st.markdown("**What each component measures (original dataset)**")
    for component in original_score.components:
        st.markdown(f"- **{component.component_name}**: {component.explanation}")


def render_save_run(
    result: LoadedDataset,
    cleaning_result: CleaningResult | None,
    original_issues: list[Issue],
    original_score: QualityScore,
    cleaned_issues: list[Issue] | None,
    cleaned_score: QualityScore | None,
) -> None:
    """Render the Save Run to Database button and a local-session-only clear option."""
    dataset_key = f"{result.file_name}:{result.dataframe.shape}:{result.sheet_name}"
    run_id = _get_or_create_run_id(cleaning_result, dataset_key)
    st.caption(f"This run's ID: `{run_id}`")

    save_col, clear_col = st.columns(2)

    if save_col.button("💾 Save Run to Database", type="primary"):
        try:
            applied_fixes_summary = None
            if cleaning_result is not None:
                affected_counts = Counter(entry.cleaning_action for entry in cleaning_result.audit_log)
                fix_titles = {fix.fix_id: fix.title for fix in SAFE_FIX_DEFINITIONS}
                applied_fixes_summary = [
                    AppliedFixSummary(fix_id, fix_titles.get(fix_id, fix_id), affected_counts.get(fix_id, 0))
                    for fix_id in cleaning_result.applied_fix_ids
                ]
            bundle = RunBundle(
                run_id=run_id,
                dataset_name=result.file_name,
                file_type=result.file_type,
                selected_sheet=result.sheet_name,
                run_timestamp=datetime.now(timezone.utc).isoformat(),
                original_shape=result.dataframe.shape,
                original_profiles=profile_dataframe(result.dataframe),
                original_issues=original_issues,
                original_score=original_score,
                cleaned_shape=cleaning_result.cleaned_df.shape if cleaning_result is not None else None,
                cleaned_profiles=profile_dataframe(cleaning_result.cleaned_df) if cleaning_result is not None else None,
                cleaned_issues=cleaned_issues,
                cleaned_score=cleaned_score,
                audit_log=cleaning_result.audit_log if cleaning_result is not None else None,
                applied_fixes=applied_fixes_summary,
            )
            save_run(bundle, DATABASE_PATH)
            st.session_state.setdefault("saved_run_ids", set()).add(run_id)
            st.success(f"Run `{run_id}` saved to the database.")
        except DatabaseError as exc:
            st.error(str(exc))

    if clear_col.button("Clear Current Unsaved Session"):
        st.session_state.pop("cleaning_result", None)
        st.session_state.pop("cleaning_dataset_key", None)
        st.session_state.pop("cleaning_decision_log", None)
        st.session_state.pop(f"original_run_id::{dataset_key}", None)
        st.info("Current session cleared. Previously saved runs in the database are not affected.")
        st.rerun()


def render_run_history() -> None:
    """Render the saved-run history table, filters, run comparison, and run detail."""
    try:
        init_db(DATABASE_PATH)
        history = get_run_history(DATABASE_PATH)
    except DatabaseError as exc:
        st.error(f"Could not load run history: {exc}")
        return

    if history.empty:
        st.info("No saved runs yet. Save a completed analysis to begin building run history.")
        return

    st.caption(f"{len(history):,} run(s) saved.")

    st.markdown("#### Filters")
    with st.container(border=True, key="card_run_history_filters"):
        fcol1, fcol2 = st.columns(2)
        dataset_filter = fcol1.multiselect("Dataset", sorted(history["dataset_name"].unique()))
        min_date = fcol2.date_input("On or after", value=None)

    filtered = history
    if dataset_filter:
        filtered = filtered[filtered["dataset_name"].isin(dataset_filter)]
    if min_date:
        filtered = filtered[pd.to_datetime(filtered["run_timestamp"]).dt.date >= min_date]

    st.markdown("#### Saved Runs")
    display_df = filtered.copy()
    display_df["run_timestamp"] = pd.to_datetime(display_df["run_timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    friendly_columns = {
        "dataset_name": "Dataset", "run_timestamp": "Date",
        "original_score": "Original Score", "cleaned_score": "Cleaned Score", "score_improvement": "Improvement",
        "original_issue_count": "Issues Before", "cleaned_issue_count": "Issues After",
    }
    expandable_table(
        display_df[list(friendly_columns)].rename(columns=friendly_columns),
        preview_rows=QUICK_PREVIEW_ROWS, key="run_history_view_all", width='stretch',
    )
    st.caption("Run IDs and database details are in Run Detail below.")

    run_options = filtered["run_id"].tolist()

    st.markdown("#### Compare Two Runs")
    if len(run_options) >= 2:
        ccol1, ccol2 = st.columns(2)
        run_a = ccol1.selectbox("Run A", run_options, index=0)
        run_b = ccol2.selectbox("Run B", run_options, index=1)
        if run_a == run_b:
            st.warning("Select two different runs to compare.")
        else:
            compare_columns = [
                "score_improvement", "original_score", "cleaned_score",
                "original_issue_count", "cleaned_issue_count",
                "dataset_name", "run_timestamp",
            ]
            row_a = filtered.loc[filtered["run_id"] == run_a, compare_columns].iloc[0]
            row_b = filtered.loc[filtered["run_id"] == run_b, compare_columns].iloc[0]
            compare_table = pd.DataFrame({"Run A": row_a, "Run B": row_b})
            compare_table.index = compare_table.index.map(lambda c: friendly_columns.get(c, c))
            st.dataframe(compare_table, width='stretch')
    else:
        st.caption("Save at least two runs to compare them.")

    st.markdown("#### Run Detail")
    selected_run = st.selectbox("Select a run to inspect", run_options)
    render_run_detail_view(selected_run)


def render_run_detail_view(run_id: str) -> None:
    """Render a detailed, expandable breakdown of one saved run."""
    try:
        detail = get_run_detail(DATABASE_PATH, run_id)
    except DatabaseError as exc:
        st.error(f"Could not load details for run '{run_id}': {exc}")
        return

    st.caption(f"Run ID: `{run_id}`")
    with st.expander("Technical database details"):
        st.dataframe(detail["run"], width='stretch', hide_index=True)

    with st.expander("Column profiles"):
        st.dataframe(detail["column_profiles"], width='stretch', hide_index=True)
    with st.expander("Quality score breakdown"):
        st.dataframe(detail["scores"], width='stretch', hide_index=True)
    with st.expander(f"Issues (showing up to {DEFAULT_MAX_ISSUES_DISPLAYED})"):
        st.dataframe(detail["issues"].head(DEFAULT_MAX_ISSUES_DISPLAYED), width='stretch', hide_index=True)
    with st.expander("Applied fixes"):
        st.dataframe(detail["applied_fixes"], width='stretch', hide_index=True)
    with st.expander(f"Cleaning audit log (showing up to {DEFAULT_MAX_ISSUES_DISPLAYED})"):
        st.dataframe(detail["audit_log"].head(DEFAULT_MAX_ISSUES_DISPLAYED), width='stretch', hide_index=True)


def render_upload_page(result: LoadedDataset | None) -> None:
    """Render the Upload page: hero + capabilities before a file loads, dataset overview after.

    Merges what used to be two separate pages (Upload Data, Dataset
    Overview) -- there is no reason to make the user visit a second page
    just to see the shape of the file they just uploaded (Phase 8.2).
    """
    if result is None:
        page_header("Data Quality Agent", "Human-guided cleaning for structured CSV and Excel datasets.")

        with st.container(border=True, key="card_upload_hero"):
            st.markdown(
                '<div class="dqa-upload-hero">'
                '<div class="dqa-upload-hero-icon">⬆</div>'
                '<div class="dqa-upload-hero-title">Upload a CSV or Excel file to get started</div>'
                '<div class="dqa-upload-hero-subtitle">Use the uploader in the sidebar ←</div>'
                "</div>",
                unsafe_allow_html=True,
            )

        ccol1, ccol2, ccol3 = st.columns(3)
        ccol1.markdown("Automatic data profiling")
        ccol2.markdown("Human-reviewed cleaning")
        ccol3.markdown("Clean CSV / Excel with audit trail")

        with st.expander("How it works"):
            st.markdown(
                """
                No external AI/LLM API is used. Every check is selected automatically from the
                dataset's inferred schema and logical column types, and every decision is
                deterministic and explainable. Domain-specific or ambiguous judgment calls (an
                uncertain column type, which values to fix) always stay under your control.
                """
            )

        st.divider()
        st.caption("Don't have a dataset?")
        if st.button("Use Demo Dataset", key="use_demo_dataset_btn"):
            st.session_state["use_demo_dataset"] = True
            st.rerun()
        return

    badge = f"{result.file_name} (demo)" if st.session_state.get("use_demo_dataset") else result.file_name
    page_header("Upload", "Understand the structure and completeness of your dataset.", badge=badge)
    if st.session_state.get("use_demo_dataset"):
        st.info("Demo dataset loaded -- this is a synthetic sample dataset, not your own data.")
    render_dataset_overview(result)
    render_preview(result)


def _render_before_after_bar(
    df: pd.DataFrame, category_col: str, empty_message: str, *, horizontal: bool = False, show_values: bool = False
) -> None:
    """Shared grouped-bar chart for a "before" (and optional "after") comparison DataFrame.

    Always uses the shared semantic palette (Before = blue, After = teal) --
    never Plotly's default color cycling -- so the same concept reads the
    same color on every chart in the app. ``show_values`` prints the number
    directly on each bar -- only used where the category count stays small
    (e.g. the five quality components), never on charts that can list many
    categories, where printed labels would just overlap and clutter.
    """
    if df.empty:
        st.info(empty_message)
        return
    value_cols = [c for c in ("before", "after") if c in df.columns]
    melted = df.melt(id_vars=category_col, value_vars=value_cols, var_name="State", value_name="Count")
    melted["State"] = melted["State"].map({"before": "Before", "after": "After"})
    text_arg = "Count" if show_values else None
    if horizontal:
        fig = px.bar(
            melted, y=category_col, x="Count", color="State", barmode="group",
            orientation="h", color_discrete_map=BEFORE_AFTER_MAP, text=text_arg,
        )
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_xaxes(title="Count")
    else:
        fig = px.bar(melted, x=category_col, y="Count", color="State", barmode="group", color_discrete_map=BEFORE_AFTER_MAP, text=text_arg)
        fig.update_layout(xaxis_title="")
    if show_values:
        fig.update_traces(textposition="outside")
    apply_chart_theme(fig)
    st.plotly_chart(fig)


def _split_accepted_vs_needs_review(unresolved_issues: list[Issue], decision_log: list[CleaningDecisionLogEntry]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split remaining issues into ones already reviewed-and-accepted vs. ones still needing review.

    A purely presentational split for the dashboard: uses the existing
    Cleaning Decision Log to know which (column, issue type) combinations a
    human already looked at this session -- it does not change detection,
    scoring, or which issues count as "unresolved."
    """
    decided_keys = {(d.column_name, d.issue_type) for d in decision_log}
    accepted = [i for i in unresolved_issues if (i.column_name, i.issue_type) in decided_keys]
    needs_review = [i for i in unresolved_issues if (i.column_name, i.issue_type) not in decided_keys]
    return unresolved_issues_summary(accepted), unresolved_issues_summary(needs_review)


def _go_to_page_button(label: str, target_page: str, *, key: str) -> None:
    """A button that jumps straight to another primary page (e.g. from an empty state)."""
    if st.button(label, key=key):
        st.session_state["nav_page"] = target_page
        st.rerun()


def render_results_page(result: LoadedDataset, cleaning_result: CleaningResult | None) -> None:
    """Render the Results page: cleaning summary, before/after dashboard, remaining issues, downloads.

    Merges what used to be two separate destinations -- the Data Quality
    Dashboard and the cleaning downloads (previously at the bottom of Data
    Cleaning) -- into the one page that answers "what changed, and how do I
    get my file?" (Phase 8.2). Works generically for any structured dataset;
    every chart is driven purely by the generic Issue/QualityScore/AuditLogEntry
    records already computed elsewhere in the app.
    """
    page_header("Results", "See what changed and download your cleaned dataset.", badge=result.file_name)

    original_df = result.dataframe
    original_detection = detect_issues(original_df, _get_effective_profiles(original_df), result.file_name)
    original_score = calculate_quality_score(original_df, original_detection.issues, "original")

    cleaned_detection = None
    cleaned_score = None
    if cleaning_result is not None:
        cleaned_df = cleaning_result.cleaned_df
        cleaned_detection = detect_issues(cleaned_df, profile_dataframe(cleaned_df), result.file_name)
        cleaned_score = calculate_quality_score(cleaned_df, cleaned_detection.issues, "cleaned")

    cleaned_summary: DetectionSummary | None = cleaned_detection.summary if cleaned_detection else None
    kpis = compute_kpis(
        original_score, original_detection.summary, cleaned_score, cleaned_summary,
        len(cleaning_result.audit_log) if cleaning_result is not None else 0,
    )

    st.markdown("#### Cleaning Summary")
    kpi_card_row(
        [
            {"label": "Original Score", "value": f"{kpis.original_score:.1f} / 100", "accent": "primary"},
            {
                "label": "Cleaned Score",
                "value": f"{kpis.cleaned_score:.1f} / 100" if kpis.cleaned_score is not None else "N/A",
                "accent": "success" if kpis.cleaned_score is not None else "neutral",
            },
            {
                "label": "Score Improvement",
                "value": f"{kpis.score_improvement:+.1f}" if kpis.score_improvement is not None else "N/A",
                "accent": "success" if (kpis.score_improvement or 0) >= 0 else "danger",
            },
        ]
    )
    kpi_card_row(
        [
            {"label": "Issues Before", "value": f"{kpis.total_issues_before:,}", "accent": "primary"},
            {"label": "Issues After", "value": f"{kpis.total_issues_after:,}" if kpis.total_issues_after is not None else "N/A", "accent": "success"},
            {"label": "Issues Resolved", "value": f"{kpis.issues_resolved:,}" if kpis.issues_resolved is not None else "N/A", "accent": "success"},
            {"label": "Values Changed", "value": f"{kpis.total_changes_applied:,}", "accent": "neutral"},
        ]
    )

    st.divider()

    if cleaning_result is None:
        with st.container(border=True, key="card_download_empty"):
            st.markdown("#### Your cleaned dataset will appear here")
            st.caption("Clean the dataset first to unlock the download and the before/after comparison.")
            _go_to_page_button("Go to Clean Data", "Clean Data", key="results_goto_clean")
    else:
        render_primary_download(result, cleaning_result)

    st.divider()
    st.markdown("#### Before vs After Dashboard")

    if cleaning_result is not None:
        grid1_left, grid1_right = st.columns(2)
        with grid1_left:
            st.markdown("**Issues Before vs After**")
            total_df = issues_total_comparison(original_detection.summary, cleaned_summary)
            fig_total = px.bar(total_df, x="state", y="issue_count", color="state", text="issue_count", color_discrete_map=BEFORE_AFTER_MAP)
            fig_total.update_layout(xaxis_title="", yaxis_title="Total issues")
            apply_chart_theme(fig_total, show_legend=False)
            st.plotly_chart(fig_total)
        with grid1_right:
            st.markdown("**Issues by Severity**")
            _render_before_after_bar(
                issues_by_severity_comparison(original_detection.summary, cleaned_summary),
                "severity", "No issues detected.", show_values=True,
            )

        grid2_left, grid2_right = st.columns(2)
        with grid2_left:
            st.markdown("**Issues by Category**")
            _render_before_after_bar(
                issues_by_category_comparison(original_detection.summary, cleaned_summary),
                "category", "No issues detected.",
            )
        with grid2_right:
            st.markdown("**Quality Components**")
            component_df = quality_component_comparison(original_score, cleaned_score)
            _render_before_after_bar(component_df, "component", "No component scores available.", show_values=True)

        with st.expander("Score details"):
            render_component_comparison(original_score, cleaned_score)
            st.markdown(_SCORE_METHODOLOGY_MARKDOWN)
            weights_df = pd.DataFrame(sorted(COMPONENT_WEIGHTS.items()), columns=["Component", "Weight (%)"])
            st.dataframe(weights_df, width='stretch', hide_index=True)

        st.markdown("**Most Problematic Columns**")
        _render_before_after_bar(
            most_problematic_columns(original_detection.issues, cleaned_detection.issues if cleaned_detection else None),
            "column_name", "No column-specific issues to rank.",
            horizontal=True,
        )

        st.markdown("**Cleaning Impact**")
        st.caption("Cleaning actions ranked by values changed.")
        cleaning_df = cleaning_impact_summary(cleaning_result.audit_log)
        if cleaning_df.empty:
            st.caption("No values changed yet.")
        else:
            ranked = cleaning_df.sort_values("affected_values", ascending=True)
            impact_fig = px.bar(ranked, y="cleaning_action", x="affected_values", orientation="h", text="affected_values")
            impact_fig.update_traces(marker_color=AFTER_COLOR, textposition="outside")
            impact_fig.update_layout(yaxis_title="", xaxis_title="Values changed")
            apply_chart_theme(impact_fig, show_legend=False)
            st.plotly_chart(impact_fig)

    st.divider()
    st.markdown("#### Remaining Issues")
    st.caption("Not necessarily errors -- accepted issues were reviewed; the rest still need a look.")
    unresolved_source = cleaned_detection.issues if cleaned_detection is not None else original_detection.issues
    unresolved_df = unresolved_issues_summary(unresolved_source)
    if unresolved_df.empty:
        st.success("No quality issues detected. Your dataset passed the current quality checks.")
    else:
        accepted_df, needs_review_df = _split_accepted_vs_needs_review(
            unresolved_source, st.session_state.get("cleaning_decision_log", [])
        )
        rename_cols = {"issue_type": "Issue type", "issue_category": "Category", "count": "Count", "example_action": "Recommended action"}

        def _humanize_and_rename(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            if "issue_type" in df.columns:
                df["issue_type"] = df["issue_type"].map(humanize_issue_type)
            return df.rename(columns=rename_cols)

        review_tab, accepted_tab = st.tabs([f"Needs Review ({int(needs_review_df['count'].sum()) if not needs_review_df.empty else 0})", f"Accepted ({int(accepted_df['count'].sum()) if not accepted_df.empty else 0})"])
        with review_tab:
            if needs_review_df.empty:
                st.success("Nothing left needing review.")
            else:
                st.dataframe(_humanize_and_rename(needs_review_df), width='stretch', hide_index=True)
        with accepted_tab:
            if accepted_df.empty:
                st.caption("No issues have been explicitly accepted yet -- use guided review on the Clean Data page.")
            else:
                st.dataframe(_humanize_and_rename(accepted_df), width='stretch', hide_index=True)

    if cleaning_result is not None:
        st.divider()
        render_secondary_downloads(result, cleaning_result)

    st.divider()
    st.markdown("#### Save This Run")
    render_save_run(
        result, cleaning_result, original_detection.issues, original_score,
        cleaned_detection.issues if cleaned_detection is not None else None, cleaned_score,
    )


def render_about_page() -> None:
    """Render the About page: summary, principles, and glossary."""
    page_header("About", "A generic, human-in-the-loop data quality agent.")

    st.markdown("#### What it does")
    st.markdown("A generic, human-in-the-loop data quality agent for structured CSV and Excel datasets.")

    st.markdown("#### How it works")
    st.markdown(" → ".join(["Upload", "Review Issues", "Clean Data", "Results"]))

    st.markdown("#### Core technologies")
    badges = " &nbsp; ".join(plain_badge(name) for name in ("Python", "Pandas", "Streamlit", "SQLite", "Plotly", "Pytest"))
    st.markdown(badges, unsafe_allow_html=True)

    st.markdown("#### Design principles")
    st.markdown(
        """
        - **Human control over ambiguous decisions.** Outliers, uncertain types, and similar
          categories are surfaced for review, never silently changed.
        - **Original data is immutable.** Cleaning always works on a separate copy.
        - **Every change is auditable.** Value changes and human decisions are logged separately.
        - **No external LLM required.** Every decision is deterministic and reproducible.
        """
    )

    with st.expander("Technical architecture"):
        st.markdown(
            """
            - `schema_inference.py` / `profiler.py` -- logical type inference and column statistics.
            - `issue_detector.py` / `rule_engine.py` -- generic, dtype-independent quality checks.
            - `cleaning_engine.py` / `cleaning_strategies.py` -- safe fixes and guided remediation.
            - `scoring.py` -- five-component weighted quality score.
            - `database.py` -- SQLite persistence of profiles, issues, and scores (never the raw data).
            """
        )

    with st.expander("Glossary"):
        for term, definition in GLOSSARY.items():
            st.markdown(f"- **{term}**: {definition}")


def main() -> None:
    """Application entry point: sidebar navigation dispatches to one page at a time."""
    page, result = render_sidebar()
    st.session_state.setdefault("visited_pages", set()).add(page)

    if page == "Upload":
        render_upload_page(result)
        return

    if result is None:
        st.warning("Upload a dataset in the sidebar first.")
        return

    if page == "Review Issues":
        render_review_issues_page(result)
    elif page == "Clean Data":
        render_data_cleaning(result)
    elif page == "Results":
        cleaning_result = st.session_state.get("cleaning_result")
        render_results_page(result, cleaning_result)
    elif page == "Run History":
        render_run_history_page()
    elif page == "About":
        render_about_page()


if __name__ == "__main__":
    main()
