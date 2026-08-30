"""Central configuration constants for the Generic AI Data Quality Agent.

Keeping these values in one place means later phases (schema inference, rule
engine, scoring) can all reference the same definitions instead of
re-declaring magic numbers or literals.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# --- Supported file types ---------------------------------------------------

CSV_EXTENSIONS: tuple[str, ...] = (".csv", ".txt")
EXCEL_EXTENSIONS: tuple[str, ...] = (".xlsx", ".xls")
SUPPORTED_EXTENSIONS: tuple[str, ...] = CSV_EXTENSIONS + EXCEL_EXTENSIONS

# --- File loading limits -----------------------------------------------------

# Reasonable ceiling for a Streamlit Community Cloud deployment; large files
# are rejected with a clear error instead of silently hanging the app.
MAX_FILE_SIZE_MB: int = 200

# Number of bytes/characters sampled when sniffing encoding and delimiter,
# so detection stays fast even on very large files.
SNIFF_SAMPLE_SIZE: int = 100_000

# --- Encoding detection -------------------------------------------------------

# Fallback encodings tried in order if automatic detection fails or produces
# low confidence. windows-1255 / cp1255 cover common Hebrew-locale exports.
ENCODING_FALLBACKS: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "windows-1255",
    "cp1255",
    "windows-1252",
    "latin-1",
)

# Below this confidence, automatic detection is not trusted and utf-8 is
# tried first instead.
ENCODING_CONFIDENCE_THRESHOLD: float = 0.5

# --- CSV delimiter detection --------------------------------------------------

CSV_DELIMITER_CANDIDATES: tuple[str, ...] = (",", ";", "\t", "|")
DEFAULT_CSV_DELIMITER: str = ","

# --- Schema inference (logical type detection) --------------------------------

# Minimum fraction of non-null values that must match a pattern for the
# column to be confidently classified as that type.
EMAIL_MATCH_THRESHOLD: float = 0.8
URL_MATCH_THRESHOLD: float = 0.8
PHONE_MATCH_THRESHOLD: float = 0.8
CURRENCY_MATCH_THRESHOLD: float = 0.7
NUMERIC_TEXT_THRESHOLD: float = 0.9

# Date detection is two-staged: a cheap regex prefilter gate, then an actual
# pd.to_datetime parse-success check on the values that passed the gate.
DATE_LIKE_GATE_THRESHOLD: float = 0.6
DATE_PARSE_SUCCESS_THRESHOLD: float = 0.85

# A column falls into "Mixed type" when the single best-matching pattern
# explains a meaningful but non-dominant share of the values -- i.e. no
# single type describes the column, but it isn't pure noise either.
MIXED_TYPE_LOWER_BOUND: float = 0.15
MIXED_TYPE_UPPER_BOUND: float = 0.85

# "Possible identifier": very high uniqueness plus a corroborating signal
# (a column-name hint or a consistent value format), never uniqueness alone.
IDENTIFIER_UNIQUE_RATIO_THRESHOLD: float = 0.95
IDENTIFIER_MIN_ROWS: int = 5
IDENTIFIER_FORMAT_CONSISTENCY_THRESHOLD: float = 0.8

# "Categorical": a small, repeating set of values relative to the column size.
CATEGORICAL_MAX_UNIQUE_VALUES: int = 30
CATEGORICAL_UNIQUE_RATIO_THRESHOLD: float = 0.2

# "Free text": long and/or highly unique string values with no other
# structure detected.
FREE_TEXT_MIN_AVG_LENGTH: float = 25.0
FREE_TEXT_UNIQUE_RATIO_THRESHOLD: float = 0.9
FREE_TEXT_MIN_AVG_LENGTH_FOR_UNIQUE_RULE: float = 8.0

# Column-name substrings used only as a weak corroborating signal, never as
# the sole basis for a type decision.
IDENTIFIER_NAME_HINTS: tuple[str, ...] = ("id", "code", "number", "key", "uuid", "guid", "identifier")
CURRENCY_NAME_HINTS: tuple[str, ...] = (
    "price", "cost", "amount", "salary", "fee", "revenue", "total",
    "pay", "wage", "income", "expense", "cash", "balance",
)
PHONE_NAME_HINTS: tuple[str, ...] = ("phone", "tel", "mobile", "cell", "fax")

# Numeric-dtype semantic override (Phase 7A correction): a numeric pandas
# dtype does not by itself guarantee the column is really a measurement --
# e.g. a "Phone" column full of digits parses as int64, but averaging or
# outlier-checking it is meaningless. A column-name hint is a *necessary but
# not sufficient* gate here (never the sole factor): it must additionally be
# corroborated by value-shape evidence (consistent digit length and/or high
# uniqueness) before the column is reclassified away from Integer. A name
# hint with no corroboration only lowers confidence -- it never forces a
# reclassification, and no reclassification happens below this minimum
# sample size (too few rows makes length/uniqueness checks unreliable).
NUMERIC_SEMANTIC_MIN_ROWS: int = 8
NUMERIC_PHONE_LENGTH_RANGE: tuple[int, int] = (7, 15)
NUMERIC_LENGTH_CONSISTENCY_THRESHOLD: float = 0.8
NUMERIC_HIGH_UNIQUENESS_THRESHOLD: float = 0.9
NUMERIC_SEMANTIC_HIGH_CONFIDENCE: float = 0.85
NUMERIC_SEMANTIC_MODERATE_CONFIDENCE: float = 0.65
NUMERIC_SEMANTIC_REDUCED_CONFIDENCE: float = 0.7

# --- Column profiling ----------------------------------------------------------

MAX_EXAMPLE_VALUES: int = 5
IQR_OUTLIER_MULTIPLIER: float = 1.5

# --- Issue detection (Phase 3) --------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MISSING_PLACEHOLDERS_PATH: Path = PROJECT_ROOT / "config" / "missing_placeholders.json"


@lru_cache(maxsize=1)
def load_missing_placeholders() -> frozenset[str]:
    """Load configurable missing-value placeholder tokens.

    Values are matched case-insensitively after stripping whitespace, so the
    JSON file only needs to list one canonical form per placeholder (e.g.
    "n/a" also matches "N/A", " N/a ").

    Returns:
        A frozenset of lowercase placeholder tokens.
    """
    with open(MISSING_PLACEHOLDERS_PATH, "r", encoding="utf-8") as handle:
        values = json.load(handle)
    return frozenset(str(v).strip().lower() for v in values)


# Severity is assigned from the proportion of affected rows using these
# (low, medium, high) cutoffs -- below "low" is Low severity, up to "medium"
# is Medium, up to "high" is High, and anything higher is Critical.
MISSING_VALUE_SEVERITY_THRESHOLDS: tuple[float, float, float] = (0.05, 0.2, 0.5)
DUPLICATE_ROW_SEVERITY_THRESHOLDS: tuple[float, float, float] = (0.02, 0.1, 0.3)
EMPTY_ROW_SEVERITY_THRESHOLDS: tuple[float, float, float] = (0.02, 0.1, 0.3)

# IQR outlier detection (reuses IQR_OUTLIER_MULTIPLIER above).
OUTLIER_DETECTION_CONFIDENCE: float = 0.7

# A date is flagged as unusually old/far in the future beyond these
# horizons -- informational only, never treated as an error.
DATE_FAR_PAST_YEARS: int = 100
DATE_FAR_FUTURE_YEARS: int = 5

# UI-side display limits only (detection itself is always run on the full
# dataset; these only bound how much gets rendered in the Streamlit table).
DEFAULT_MAX_ISSUES_DISPLAYED: int = 500
MAX_SAMPLE_ROWS_TO_INSPECT: int = 50

# --- Data quality scoring (Phase 5) ----------------------------------------------

# Component weights (must sum to 100). Completeness and Validity are
# weighted highest because missing and outright invalid data are the most
# fundamental blockers to trusting a dataset; Uniqueness, Consistency, and
# Structural Quality are typically less severe (formatting/structure rather
# than data being wrong or absent).
COMPONENT_WEIGHTS: dict[str, float] = {
    "Completeness": 30.0,
    "Uniqueness": 15.0,
    "Validity": 25.0,
    "Consistency": 15.0,
    "Structural Quality": 15.0,
}

# Which Phase 3 issue_types feed into which score component. Every set is
# disjoint from every other -- each issue_type counts toward exactly one
# component (or none, if intentionally excluded below), so no single
# detected issue is ever penalized twice.
COMPLETENESS_ISSUE_TYPES: frozenset[str] = frozenset(
    {"missing_null", "blank_string", "whitespace_only_string", "missing_placeholder"}
)
UNIQUENESS_ISSUE_TYPES: frozenset[str] = frozenset({"exact_duplicate_row", "identifier_duplicate_value"})
VALIDITY_ISSUE_TYPES: frozenset[str] = frozenset(
    {"value_fails_type_conversion", "unexpected_text_in_numeric_column", "invalid_date", "infinite_value"}
)
CONSISTENCY_ISSUE_TYPES: frozenset[str] = frozenset(
    {
        "leading_trailing_whitespace", "repeated_internal_spaces", "non_printable_characters",
        "similar_category_values", "inconsistent_capitalization", "mixed_date_formats",
        "numeric_format_inconsistency",
    }
)
STRUCTURAL_ISSUE_TYPES: frozenset[str] = frozenset(
    {
        "empty_row", "empty_column", "duplicate_column_name", "near_duplicate_column_name",
        "column_name_whitespace", "inconsistent_column_name_formatting",
        "numeric_stored_as_text", "date_stored_as_text", "mixed_data_types",
    }
)

# Intentionally excluded from every component: these are informational /
# "review, don't assume it's wrong" findings (IQR outliers, constant
# columns, unusual-but-plausible dates), or -- for identifier issues --
# already fully reflected elsewhere (a missing identifier value is already
# counted once via Completeness's missing_null; double-counting it under a
# second component would violate "avoid double-counting the same issue").
SCORING_EXCLUDED_ISSUE_TYPES: frozenset[str] = frozenset(
    {
        "possible_outlier", "suspiciously_constant_column", "unusually_old_date",
        "unusually_future_date", "identifier_missing_values", "identifier_inconsistent_format",
        # Phase 7B: negative values are never assumed to be errors (a refund,
        # an adjustment, a temperature below zero can all be legitimate) --
        # detected for visibility in the Guided Issue Review workflow only,
        # never penalized by default. The score only ever reflects the
        # numeric issue types a user has *not yet reviewed* -- this one is
        # deliberately excluded regardless of review state.
        "negative_value",
    }
)

# --- Bundled demo dataset (portfolio/live-demo mode) ------------------------------

# A small, fully synthetic employee dataset shipped with the repo so a
# recruiter can try the app without uploading their own file. Loaded through
# the exact same src.file_loader.load_csv() pipeline as any real upload --
# never a separate demo code path.
DEMO_DATASET_PATH: Path = PROJECT_ROOT / "data" / "demo_employee_data.csv"

# --- SQLite persistence (Phase 5) -------------------------------------------------

DATABASE_PATH: Path = PROJECT_ROOT / "database" / "data_quality.db"
CREATE_TABLES_SQL_PATH: Path = PROJECT_ROOT / "sql" / "create_tables.sql"
# Bumped from 1.0.0 because Phase 6 adds column_profiles.outlier_count via an
# idempotent migration (see database.py:_apply_migrations) -- existing
# databases created under 1.0.0 are upgraded in place, not recreated.
SCHEMA_VERSION: str = "1.1.0"

# --- Human-in-the-loop logical type override (Phase 7A) ---------------------------

# Below this confidence, the Column Profiling page warns the user to review
# the inferred type before relying on it for type-specific issue detection.
TYPE_OVERRIDE_CONFIDENCE_WARNING_THRESHOLD: float = 0.7
