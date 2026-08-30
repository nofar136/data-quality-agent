"""Generic data quality issue detectors (Phase 3).

Each function here detects one category of issue (dataset-, missing-value-,
type-, text-, numeric-, date-, or identifier-level) and returns a flat list
of ``Issue`` records. ``src/rule_engine.py`` decides *which* of these to run
for a given column, based on its inferred logical type; this module only
implements *how* each check works.

Detection always runs over the full dataset -- nothing here samples or caps
results. Keeping the interface to Streamlit manageable for very large
result sets is a UI-layer concern (see app.py), not a detection concern, so
that headline counts are always exact.

Severity policy
----------------
Severity is never random. Two mechanisms are used, both deterministic:

1. **Ratio-based**: for issues defined by a proportion of affected rows
   (missing values, duplicate rows, empty rows), severity climbs through
   Low -> Medium -> High -> Critical as that proportion crosses the
   thresholds in ``src/config.py`` (``*_SEVERITY_THRESHOLDS``).
2. **Fixed-by-type**: structural or semantic issues (an empty column, a
   mixed-type column, an infinite value, a duplicate column name) have a
   severity chosen once, based on how disruptive that class of issue
   typically is to downstream analysis -- documented inline at each call
   site below.

Values that are legitimately ambiguous (negative numbers, unusual-but-valid
dates, duplicate identifiers, statistical outliers) are always capped at Low
or Medium and their recommended action explicitly asks for review rather
than asserting an error, per the project's "never assume, always flag"
principle.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    DATE_FAR_FUTURE_YEARS,
    DATE_FAR_PAST_YEARS,
    DUPLICATE_ROW_SEVERITY_THRESHOLDS,
    EMPTY_ROW_SEVERITY_THRESHOLDS,
    IQR_OUTLIER_MULTIPLIER,
    MISSING_VALUE_SEVERITY_THRESHOLDS,
    OUTLIER_DETECTION_CONFIDENCE,
    load_missing_placeholders,
)
from src.models import Issue, IssueCategory, Severity, max_severity
from src.profiler import ColumnProfile, date_values, numeric_values
from src.schema_inference import (
    LogicalType,
    format_signature,
    is_numeric_string,
    normalize_column_name,
)


ISSUE_TYPE_EXPLANATIONS: dict[str, str] = {
    "exact_duplicate_row": "This row is an exact copy of another row in the dataset.",
    "empty_row": "Every column in this row is empty.",
    "empty_column": "Every value in this column is empty.",
    "duplicate_column_name": "Two or more columns have the exact same name.",
    "near_duplicate_column_name": "Two or more column names normalize to the same thing (case/spacing/punctuation aside) and may represent the same field.",
    "column_name_whitespace": "The column name has leading or trailing spaces.",
    "inconsistent_column_name_formatting": "This dataset mixes naming conventions across columns (e.g. snake_case and Title Case together).",
    "missing_null": "The value is genuinely missing (null).",
    "blank_string": "The value is an empty string.",
    "whitespace_only_string": "The value contains only whitespace characters.",
    "missing_placeholder": "The value is a recognized missing-value placeholder (e.g. 'N/A', 'unknown').",
    "numeric_stored_as_text": "This column's values are numeric but stored as text.",
    "date_stored_as_text": "This column's values are dates but stored as text.",
    "mixed_data_types": "This column contains an inconsistent mix of value types.",
    "value_fails_type_conversion": "This value could not be converted to the column's dominant type.",
    "unexpected_text_in_numeric_column": "This column is mostly numeric, but this value is non-numeric text.",
    "leading_trailing_whitespace": "The value has leading or trailing spaces.",
    "repeated_internal_spaces": "The value contains repeated spaces in the middle of the text.",
    "non_printable_characters": "The value contains non-printable/control characters.",
    "similar_category_values": "This value likely represents the same category as another value, differing only in case, spacing, punctuation, or separators.",
    "inconsistent_capitalization": "This value differs from the column's most common variant only in letter case.",
    "possible_outlier": "This value is a statistical outlier relative to the rest of the column (IQR method).",
    "negative_value": "This value is negative -- not assumed to be an error; negative values may be perfectly valid for this column (e.g. a refund or an adjustment). Review before deciding whether to treat it as invalid.",
    "infinite_value": "This value is infinite, which is never a valid numeric value.",
    "suspiciously_constant_column": "Every value in this column is identical.",
    "numeric_format_inconsistency": "This value uses a different decimal/thousands-separator convention than most of the column.",
    "invalid_date": "This value could not be parsed as a date at all.",
    "mixed_date_formats": "This date uses a different format/shape than most dates in the column.",
    "unusually_old_date": "This date is far in the past relative to typical values -- not assumed incorrect.",
    "unusually_future_date": "This date is far in the future relative to typical values -- not assumed incorrect.",
    "identifier_missing_values": "This identifier column has missing values, which is unusual for a key/ID field.",
    "identifier_duplicate_value": "This identifier value appears more than once -- not assumed to be an error.",
    "identifier_inconsistent_format": "This identifier's format differs from the majority of values in the column.",
}


# --- Shared helpers ----------------------------------------------------------------


def _make_issue(
    run_id: str,
    dataset_name: str,
    detected_at: str,
    *,
    category: str,
    issue_type: str,
    severity: str,
    confidence: float,
    recommended_action: str,
    rule_name: str,
    column_name: Optional[str] = None,
    row_index: Optional[int] = None,
    current_value: Optional[str] = None,
    suggested_value: Optional[str] = None,
    safe_to_auto_fix: bool = False,
) -> Issue:
    return Issue(
        run_id=run_id,
        dataset_name=dataset_name,
        detected_at=detected_at,
        issue_category=category,
        issue_type=issue_type,
        severity=severity,
        confidence=round(confidence, 3),
        recommended_action=recommended_action,
        rule_name=rule_name,
        column_name=column_name,
        row_index=None if row_index is None else int(row_index),
        current_value=current_value,
        suggested_value=suggested_value,
        safe_to_auto_fix=safe_to_auto_fix,
    )


def _ratio_severity(ratio: float, thresholds: tuple[float, float, float]) -> str:
    """Map an affected-row ratio to a severity using ascending thresholds."""
    low, medium, high = thresholds
    if ratio < low:
        return Severity.LOW.value
    if ratio < medium:
        return Severity.MEDIUM.value
    if ratio < high:
        return Severity.HIGH.value
    return Severity.CRITICAL.value


def _column_name_style(name: str) -> str:
    """Classify a column name's naming convention."""
    stripped = name.strip()
    if re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", stripped):
        return "snake_case"
    if re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", stripped):
        return "kebab-case"
    if " " in stripped and re.fullmatch(r"[A-Za-z0-9]+( [A-Za-z0-9]+)*", stripped):
        return "space separated"
    if re.fullmatch(r"[A-Za-z][a-zA-Z0-9]*", stripped) and any(c.isupper() for c in stripped[1:]):
        return "camelCase/PascalCase"
    return "unknown"


def _date_format_shape(value: str) -> str:
    """Coarse shape signature for a date string (digit-runs -> 'D')."""
    return re.sub(r"\d+", lambda m: "D" * len(m.group()), value)


# --- 1. Dataset-level issues ---------------------------------------------------------


def detect_dataset_level_issues(df: pd.DataFrame, dataset_name: str, run_id: str, detected_at: str) -> list[Issue]:
    """Detect structural issues that concern the dataset or its columns as a whole."""
    issues: list[Issue] = []
    total_rows = len(df)

    # Exact duplicate rows (each repeat after the first occurrence).
    duplicate_mask = df.duplicated(keep="first")
    dup_count = int(duplicate_mask.sum())
    if dup_count:
        severity = _ratio_severity(dup_count / total_rows if total_rows else 0.0, DUPLICATE_ROW_SEVERITY_THRESHOLDS)
        for idx in df.index[duplicate_mask]:
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.DATASET.value, issue_type="exact_duplicate_row",
                    severity=severity, confidence=1.0, row_index=idx,
                    recommended_action="Review and consider removing exact duplicate rows (only after explicit approval).",
                    rule_name="exact_duplicate_row", safe_to_auto_fix=True,
                )
            )

    # Completely empty rows (every column null).
    empty_row_mask = df.isna().all(axis=1)
    empty_row_count = int(empty_row_mask.sum())
    if empty_row_count:
        severity = _ratio_severity(empty_row_count / total_rows if total_rows else 0.0, EMPTY_ROW_SEVERITY_THRESHOLDS)
        for idx in df.index[empty_row_mask]:
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.DATASET.value, issue_type="empty_row",
                    severity=severity, confidence=1.0, row_index=idx,
                    recommended_action="Safe to remove -- this row has no data in any column.",
                    rule_name="empty_row", safe_to_auto_fix=True,
                )
            )

    # Completely empty columns. Fixed High severity: an unusable column is a
    # significant structural gap, though trivially fixable once confirmed.
    # Uses positional access (iloc) because df[col] returns a DataFrame, not
    # a Series, when column labels are duplicated.
    for position, col in enumerate(df.columns):
        if df.iloc[:, position].isna().all():
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.DATASET.value, issue_type="empty_column",
                    severity=Severity.HIGH.value, confidence=1.0, column_name=str(col),
                    recommended_action="Safe to remove -- this column has no data.",
                    rule_name="empty_column", safe_to_auto_fix=True,
                )
            )

    # Duplicate / near-duplicate column names: columns whose *normalized*
    # names collide. True exact duplicates should not occur past the Phase 1
    # loader (which already renames them), but are still checked here so
    # this detector is correct standalone -- fixed Critical severity because
    # ambiguous column references can silently corrupt downstream analysis.
    normalized_map: dict[str, list[str]] = {}
    for col in df.columns:
        normalized_map.setdefault(normalize_column_name(str(col)), []).append(str(col))
    for originals in normalized_map.values():
        if len(originals) > 1:
            is_exact_duplicate = len(set(originals)) < len(originals)
            issue_type = "duplicate_column_name" if is_exact_duplicate else "near_duplicate_column_name"
            severity = Severity.CRITICAL.value if is_exact_duplicate else Severity.MEDIUM.value
            for col in originals:
                issues.append(
                    _make_issue(
                        run_id, dataset_name, detected_at,
                        category=IssueCategory.DATASET.value, issue_type=issue_type,
                        severity=severity, confidence=1.0, column_name=col,
                        current_value=", ".join(originals),
                        recommended_action="Review -- these column names normalize to the same name and may represent the same field.",
                        rule_name=issue_type,
                    )
                )

    # Column names with leading/trailing whitespace.
    for col in df.columns:
        name = str(col)
        if name != name.strip():
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.DATASET.value, issue_type="column_name_whitespace",
                    severity=Severity.LOW.value, confidence=1.0, column_name=name,
                    suggested_value=name.strip(),
                    recommended_action="Trim leading/trailing spaces from the column name.",
                    rule_name="column_name_whitespace", safe_to_auto_fix=True,
                )
            )

    # Inconsistent column-name formatting: more than one naming convention in use.
    styles = {str(col): _column_name_style(str(col)) for col in df.columns}
    distinct_styles = {style for style in styles.values() if style != "unknown"}
    if len(distinct_styles) > 1:
        for col, style in styles.items():
            if style != "unknown":
                issues.append(
                    _make_issue(
                        run_id, dataset_name, detected_at,
                        category=IssueCategory.DATASET.value, issue_type="inconsistent_column_name_formatting",
                        severity=Severity.LOW.value, confidence=0.7, column_name=col, current_value=style,
                        recommended_action="Standardize all column names to a single naming convention (e.g. snake_case).",
                        rule_name="inconsistent_column_name_formatting",
                    )
                )

    return issues


# --- 2. Missing-value issues ---------------------------------------------------------


def detect_missing_value_issues(series: pd.Series, profile: ColumnProfile, dataset_name: str, run_id: str, detected_at: str) -> list[Issue]:
    """Detect actual nulls, blank/whitespace-only strings, and known placeholders.

    Zero, False, and other legitimate falsy-looking values are never treated
    as missing -- only real NaN, empty/whitespace strings, and values that
    exactly match a configured placeholder token (config/missing_placeholders.json).
    """
    issues: list[Issue] = []
    column = profile.original_name
    total = len(series)
    if total == 0:
        return issues

    null_mask = series.isna()
    null_count = int(null_mask.sum())
    if null_count:
        severity = _ratio_severity(null_count / total, MISSING_VALUE_SEVERITY_THRESHOLDS)
        for idx in series.index[null_mask]:
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.MISSING_VALUE.value, issue_type="missing_null",
                    severity=severity, confidence=1.0, column_name=column, row_index=idx,
                    recommended_action="Review completeness; filling missing values requires an explicit imputation decision.",
                    rule_name="missing_null", safe_to_auto_fix=False,
                )
            )

    if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
        return issues

    non_null = series.dropna()
    str_values = non_null.astype(str)
    stripped = str_values.str.strip()

    exact_blank_mask = str_values == ""
    whitespace_only_mask = (stripped == "") & ~exact_blank_mask

    if exact_blank_mask.any():
        count = int(exact_blank_mask.sum())
        severity = _ratio_severity(count / total, MISSING_VALUE_SEVERITY_THRESHOLDS)
        for idx in non_null.index[exact_blank_mask]:
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.MISSING_VALUE.value, issue_type="blank_string",
                    severity=severity, confidence=1.0, column_name=column, row_index=idx,
                    current_value="", suggested_value=None,
                    recommended_action="Safe to convert to a proper missing value.",
                    rule_name="blank_string", safe_to_auto_fix=True,
                )
            )

    if whitespace_only_mask.any():
        count = int(whitespace_only_mask.sum())
        severity = _ratio_severity(count / total, MISSING_VALUE_SEVERITY_THRESHOLDS)
        for idx in non_null.index[whitespace_only_mask]:
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.MISSING_VALUE.value, issue_type="whitespace_only_string",
                    severity=severity, confidence=1.0, column_name=column, row_index=idx,
                    current_value=repr(str_values.loc[idx]), suggested_value=None,
                    recommended_action="Safe to convert to a proper missing value.",
                    rule_name="whitespace_only_string", safe_to_auto_fix=True,
                )
            )

    placeholders = load_missing_placeholders()
    non_blank_mask = ~exact_blank_mask & ~whitespace_only_mask
    placeholder_mask = non_blank_mask & stripped.str.lower().isin(placeholders)
    if placeholder_mask.any():
        count = int(placeholder_mask.sum())
        severity = _ratio_severity(count / total, MISSING_VALUE_SEVERITY_THRESHOLDS)
        for idx in non_null.index[placeholder_mask]:
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.MISSING_VALUE.value, issue_type="missing_placeholder",
                    severity=severity, confidence=0.9, column_name=column, row_index=idx,
                    current_value=str_values.loc[idx], suggested_value=None,
                    recommended_action="Recognized missing-value placeholder -- safe to convert to a proper missing value.",
                    rule_name="missing_placeholder", safe_to_auto_fix=True,
                )
            )

    return issues


# --- 3. Type issues --------------------------------------------------------------------


def detect_type_issues(series: pd.Series, profile: ColumnProfile, dataset_name: str, run_id: str, detected_at: str) -> list[Issue]:
    """Detect column-level type problems and individual conversion failures."""
    issues: list[Issue] = []
    column = profile.original_name
    logical_type = profile.effective_logical_type
    non_null = series.dropna()

    if logical_type == LogicalType.NUMERIC_TEXT.value:
        issues.append(
            _make_issue(
                run_id, dataset_name, detected_at,
                category=IssueCategory.TYPE.value, issue_type="numeric_stored_as_text",
                severity=Severity.MEDIUM.value, confidence=profile.confidence, column_name=column,
                recommended_action="Convert this column to a numeric dtype.",
                rule_name="numeric_stored_as_text", safe_to_auto_fix=True,
            )
        )
        stripped = non_null.astype(str).str.strip()
        fails_mask = ~stripped.map(is_numeric_string)
        for idx in stripped.index[fails_mask]:
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.TYPE.value, issue_type="value_fails_type_conversion",
                    severity=Severity.MEDIUM.value, confidence=0.9, column_name=column, row_index=idx,
                    current_value=stripped.loc[idx],
                    recommended_action="Manual review required -- value could not be converted to this column's dominant numeric type.",
                    rule_name="value_fails_type_conversion",
                )
            )

    elif logical_type == LogicalType.DATE_TEXT.value:
        issues.append(
            _make_issue(
                run_id, dataset_name, detected_at,
                category=IssueCategory.TYPE.value, issue_type="date_stored_as_text",
                severity=Severity.MEDIUM.value, confidence=profile.confidence, column_name=column,
                recommended_action="Convert this column to a datetime dtype.",
                rule_name="date_stored_as_text", safe_to_auto_fix=True,
            )
        )
        # Individual invalid values are reported under Date Issues (see
        # detect_date_issues) to avoid flagging the same row twice.

    elif logical_type == LogicalType.MIXED.value:
        issues.append(
            _make_issue(
                run_id, dataset_name, detected_at,
                category=IssueCategory.TYPE.value, issue_type="mixed_data_types",
                severity=Severity.HIGH.value, confidence=profile.confidence, column_name=column,
                recommended_action="Review this column -- it contains a mix of incompatible value types.",
                rule_name="mixed_data_types",
            )
        )
        # If the values are predominantly numeric with a minority of stray
        # text, call those out specifically as unexpected text.
        dominant = profile.evidence.get("dominant_partial_type")
        if dominant == "numeric":
            stripped = non_null.astype(str).str.strip()
            unexpected_mask = ~stripped.map(is_numeric_string) & (stripped != "")
            for idx in stripped.index[unexpected_mask]:
                issues.append(
                    _make_issue(
                        run_id, dataset_name, detected_at,
                        category=IssueCategory.TYPE.value, issue_type="unexpected_text_in_numeric_column",
                        severity=Severity.HIGH.value, confidence=0.7, column_name=column, row_index=idx,
                        current_value=stripped.loc[idx],
                        recommended_action="This column is mostly numeric -- review this non-numeric value.",
                        rule_name="unexpected_text_in_numeric_column",
                    )
                )

    return issues


# --- 4. Text-quality issues -------------------------------------------------------------


def _capitalization_variants(str_values: pd.Series) -> list[tuple[int, str, str]]:
    """Rows whose only difference from the column's most common variant is letter case."""
    results: list[tuple[int, str, str]] = []
    case_key = str_values.str.strip().str.lower()
    for _, group in str_values.groupby(case_key):
        variants = group.unique()
        if len(variants) <= 1:
            continue
        canonical = group.value_counts().idxmax()
        for idx, value in group.items():
            if value != canonical:
                results.append((idx, value, canonical))
    return results


def _similar_category_variants(str_values: pd.Series) -> tuple[list[tuple[int, str, str]], set]:
    """Rows that match another value once case/spacing/punctuation/hyphens/underscores
    are ignored, but differ by more than letter case alone.

    Returns the issues plus the set of row indices covered, so a separate
    pure-case-only check can skip them and avoid double-flagging.
    """
    results: list[tuple[int, str, str]] = []
    covered: set = set()
    aggressive_key = str_values.str.strip().str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    counts = str_values.value_counts()
    for key, group in str_values.groupby(aggressive_key):
        if key == "":
            continue
        variants = group.unique()
        if len(variants) <= 1:
            continue
        case_keys = {v.strip().lower() for v in variants}
        if len(case_keys) <= 1:
            continue  # pure case-only difference -- left to the capitalization check
        canonical = max(variants, key=lambda v: counts.get(v, 0))
        for idx, value in group.items():
            covered.add(idx)
            if value != canonical:
                results.append((idx, value, canonical))
    return results, covered


def detect_text_issues(series: pd.Series, profile: ColumnProfile, dataset_name: str, run_id: str, detected_at: str) -> list[Issue]:
    """Detect whitespace, capitalization, non-printable characters, and near-duplicate categories."""
    issues: list[Issue] = []
    column = profile.original_name
    non_null = series.dropna()
    if non_null.empty:
        return issues

    str_values = non_null.astype(str)

    lt_mask = str_values != str_values.str.strip()
    for idx, value in str_values[lt_mask].items():
        issues.append(
            _make_issue(
                run_id, dataset_name, detected_at,
                category=IssueCategory.TEXT.value, issue_type="leading_trailing_whitespace",
                severity=Severity.LOW.value, confidence=1.0, column_name=column, row_index=idx,
                current_value=value, suggested_value=value.strip(),
                recommended_action="Trim leading/trailing whitespace.",
                rule_name="leading_trailing_whitespace", safe_to_auto_fix=True,
            )
        )

    # Checked on the stripped value so a leading/trailing run of spaces
    # (already reported as leading_trailing_whitespace) isn't also counted
    # here -- only whitespace runs strictly between non-whitespace characters
    # are "internal".
    repeated_space_mask = str_values.str.strip().str.contains(r"\s{2,}", regex=True)
    for idx, value in str_values[repeated_space_mask].items():
        issues.append(
            _make_issue(
                run_id, dataset_name, detected_at,
                category=IssueCategory.TEXT.value, issue_type="repeated_internal_spaces",
                severity=Severity.LOW.value, confidence=1.0, column_name=column, row_index=idx,
                current_value=value, suggested_value=re.sub(r"\s+", " ", value).strip(),
                recommended_action="Collapse repeated internal spaces into a single space.",
                rule_name="repeated_internal_spaces", safe_to_auto_fix=True,
            )
        )

    def _has_non_printable(value: str) -> bool:
        return any((not ch.isprintable()) and ch not in ("\n", "\t") for ch in value)

    non_printable_mask = str_values.map(_has_non_printable)
    for idx, value in str_values[non_printable_mask].items():
        cleaned = "".join(ch for ch in value if ch.isprintable() or ch in ("\n", "\t"))
        issues.append(
            _make_issue(
                run_id, dataset_name, detected_at,
                category=IssueCategory.TEXT.value, issue_type="non_printable_characters",
                severity=Severity.MEDIUM.value, confidence=1.0, column_name=column, row_index=idx,
                current_value=repr(value), suggested_value=cleaned,
                recommended_action="Remove non-printable characters.",
                rule_name="non_printable_characters", safe_to_auto_fix=True,
            )
        )

    covered_by_similarity: set = set()
    if profile.effective_logical_type == LogicalType.CATEGORICAL.value:
        similar, covered_by_similarity = _similar_category_variants(str_values)
        for idx, value, canonical in similar:
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.TEXT.value, issue_type="similar_category_values",
                    severity=Severity.MEDIUM.value, confidence=0.75, column_name=column, row_index=idx,
                    current_value=value, suggested_value=canonical,
                    recommended_action="This value likely represents the same category as others in this column -- review before merging.",
                    rule_name="similar_category_values",
                )
            )

    if profile.effective_logical_type in (LogicalType.CATEGORICAL.value, LogicalType.IDENTIFIER.value, LogicalType.FREE_TEXT.value):
        for idx, value, canonical in _capitalization_variants(str_values):
            if idx in covered_by_similarity:
                continue
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.TEXT.value, issue_type="inconsistent_capitalization",
                    severity=Severity.LOW.value, confidence=0.8, column_name=column, row_index=idx,
                    current_value=value, suggested_value=canonical,
                    recommended_action="Standardize capitalization to match the most common variant.",
                    rule_name="inconsistent_capitalization",
                )
            )

    return issues


# --- 5. Numeric issues ------------------------------------------------------------------


def _numeric_format_inconsistency(series: pd.Series, profile: ColumnProfile, dataset_name: str, run_id: str, detected_at: str) -> list[Issue]:
    """Flag values using a minority decimal-separator convention (best-effort heuristic)."""
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return []

    comma_decimal_mask = non_null.str.match(r"^-?\d{1,3}(\.\d{3})*,\d+$")
    dot_decimal_mask = non_null.str.match(r"^-?\d{1,3}(,\d{3})*\.\d+$")
    comma_count, dot_count = int(comma_decimal_mask.sum()), int(dot_decimal_mask.sum())
    if not (comma_count and dot_count):
        return []

    if comma_count >= dot_count:
        minority_mask, majority_style = dot_decimal_mask, "comma-decimal (e.g. 1.234,56)"
    else:
        minority_mask, majority_style = comma_decimal_mask, "dot-decimal (e.g. 1,234.56)"

    issues = []
    for idx in non_null.index[minority_mask]:
        issues.append(
            _make_issue(
                run_id, dataset_name, detected_at,
                category=IssueCategory.NUMERIC.value, issue_type="numeric_format_inconsistency",
                severity=Severity.MEDIUM.value, confidence=0.6, column_name=profile.original_name, row_index=idx,
                current_value=non_null.loc[idx],
                recommended_action=f"Most values in this column use {majority_style} formatting -- standardize this value to match.",
                rule_name="numeric_format_inconsistency",
            )
        )
    return issues


def detect_numeric_issues(series: pd.Series, profile: ColumnProfile, dataset_name: str, run_id: str, detected_at: str) -> list[Issue]:
    """Detect outliers, infinite values, format inconsistencies, and constant columns.

    Negative values are never flagged as invalid on their own -- only
    genuine statistical outliers (via IQR) or infinities are reported, and
    both are capped at Low/Critical with a "review, don't assume" wording.
    """
    issues: list[Issue] = []
    column = profile.original_name
    logical_type = profile.effective_logical_type

    numeric = numeric_values(series, logical_type).dropna()
    if numeric.empty:
        return issues

    inf_mask = np.isinf(numeric.to_numpy(dtype=float))
    if inf_mask.any():
        for idx in numeric.index[inf_mask]:
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.NUMERIC.value, issue_type="infinite_value",
                    severity=Severity.CRITICAL.value, confidence=1.0, column_name=column, row_index=idx,
                    current_value=str(series.loc[idx]),
                    recommended_action="Investigate the source calculation -- infinite values are never valid.",
                    rule_name="infinite_value",
                )
            )

    finite = numeric[~inf_mask]

    if len(finite) >= 4:
        q1, q3 = finite.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            lower, upper = q1 - IQR_OUTLIER_MULTIPLIER * iqr, q3 + IQR_OUTLIER_MULTIPLIER * iqr
            outlier_mask = (finite < lower) | (finite > upper)
            for idx in finite.index[outlier_mask]:
                issues.append(
                    _make_issue(
                        run_id, dataset_name, detected_at,
                        category=IssueCategory.NUMERIC.value, issue_type="possible_outlier",
                        severity=Severity.LOW.value, confidence=OUTLIER_DETECTION_CONFIDENCE, column_name=column, row_index=idx,
                        current_value=str(series.loc[idx]),
                        recommended_action="Statistical outlier via the IQR method -- review, do not assume it is an error.",
                        rule_name="possible_outlier_iqr",
                    )
                )

    negative_mask = finite < 0
    if negative_mask.any():
        for idx in finite.index[negative_mask]:
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.NUMERIC.value, issue_type="negative_value",
                    severity=Severity.LOW.value, confidence=1.0, column_name=column, row_index=idx,
                    current_value=str(series.loc[idx]),
                    recommended_action="Negative values may be valid for this column -- review before deciding whether to treat them as invalid.",
                    rule_name="negative_value",
                )
            )

    if finite.nunique() == 1 and len(finite) >= 2:
        issues.append(
            _make_issue(
                run_id, dataset_name, detected_at,
                category=IssueCategory.NUMERIC.value, issue_type="suspiciously_constant_column",
                severity=Severity.LOW.value, confidence=0.6, column_name=column,
                current_value=str(finite.iloc[0]),
                recommended_action="Verify this is expected -- every value in this column is identical.",
                rule_name="suspiciously_constant_column",
            )
        )

    if logical_type in (LogicalType.NUMERIC_TEXT.value, LogicalType.CURRENCY.value):
        issues.extend(_numeric_format_inconsistency(series, profile, dataset_name, run_id, detected_at))

    return issues


# --- 6. Date issues ---------------------------------------------------------------------


def detect_date_issues(series: pd.Series, profile: ColumnProfile, dataset_name: str, run_id: str, detected_at: str) -> list[Issue]:
    """Detect invalid dates, mixed formats, and unusually old/future dates.

    Unusual dates are always Low severity and explicitly framed as
    "review, don't assume incorrect" -- they are informational, not errors.
    """
    issues: list[Issue] = []
    column = profile.original_name
    logical_type = profile.effective_logical_type
    non_null_raw = series.dropna()
    dates = date_values(series, logical_type)

    if logical_type == LogicalType.DATE_TEXT.value:
        invalid_mask = dates.isna()
        raw_strings = non_null_raw.astype(str).str.strip()

        for idx in raw_strings.index[invalid_mask]:
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.DATE.value, issue_type="invalid_date",
                    severity=Severity.HIGH.value, confidence=0.9, column_name=column, row_index=idx,
                    current_value=raw_strings.loc[idx],
                    recommended_action="Value could not be parsed as a date -- review manually.",
                    rule_name="invalid_date",
                )
            )

        valid_strings = raw_strings[~invalid_mask]
        if not valid_strings.empty:
            shapes = valid_strings.map(_date_format_shape)
            shape_counts = shapes.value_counts()
            if len(shape_counts) > 1:
                majority_shape = shape_counts.idxmax()
                for idx, shape in shapes.items():
                    if shape != majority_shape:
                        issues.append(
                            _make_issue(
                                run_id, dataset_name, detected_at,
                                category=IssueCategory.DATE.value, issue_type="mixed_date_formats",
                                severity=Severity.MEDIUM.value, confidence=0.7, column_name=column, row_index=idx,
                                current_value=valid_strings.loc[idx],
                                recommended_action=f"Most dates in this column use the '{majority_shape}' shape -- standardize this value to match.",
                                rule_name="mixed_date_formats",
                            )
                        )

    valid_dates = dates.dropna()
    if not valid_dates.empty:
        now = pd.Timestamp.now(tz=valid_dates.dt.tz)
        far_past_cutoff = now - pd.DateOffset(years=DATE_FAR_PAST_YEARS)
        far_future_cutoff = now + pd.DateOffset(years=DATE_FAR_FUTURE_YEARS)

        for idx in valid_dates.index[valid_dates < far_past_cutoff]:
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.DATE.value, issue_type="unusually_old_date",
                    severity=Severity.LOW.value, confidence=0.5, column_name=column, row_index=idx,
                    current_value=str(valid_dates.loc[idx]),
                    recommended_action="Unusually old date -- review, but do not assume it is incorrect.",
                    rule_name="unusually_old_date",
                )
            )
        for idx in valid_dates.index[valid_dates > far_future_cutoff]:
            issues.append(
                _make_issue(
                    run_id, dataset_name, detected_at,
                    category=IssueCategory.DATE.value, issue_type="unusually_future_date",
                    severity=Severity.LOW.value, confidence=0.5, column_name=column, row_index=idx,
                    current_value=str(valid_dates.loc[idx]),
                    recommended_action="Unusually far-future date -- review, but do not assume it is incorrect.",
                    rule_name="unusually_future_date",
                )
            )

    return issues


# --- 7. Identifier issues -----------------------------------------------------------------


def detect_identifier_issues(series: pd.Series, profile: ColumnProfile, dataset_name: str, run_id: str, detected_at: str) -> list[Issue]:
    """Detect missing, duplicate, and inconsistently formatted identifier values.

    Duplicate identifiers are never assumed to be errors -- the recommended
    action always asks for review (e.g. it may be a legitimate repeat/update).
    """
    issues: list[Issue] = []
    column = profile.original_name
    total = len(series)
    non_null = series.dropna()

    missing_count = total - len(non_null)
    if missing_count and total:
        ratio = missing_count / total
        severity = max_severity(_ratio_severity(ratio, MISSING_VALUE_SEVERITY_THRESHOLDS), Severity.MEDIUM.value)
        issues.append(
            _make_issue(
                run_id, dataset_name, detected_at,
                category=IssueCategory.IDENTIFIER.value, issue_type="identifier_missing_values",
                severity=severity, confidence=1.0, column_name=column,
                current_value=f"{missing_count} missing value(s) ({round(ratio * 100, 2)}%)",
                recommended_action="Identifiers should generally not be missing -- review before using this column as a key.",
                rule_name="identifier_missing_values",
            )
        )

    str_values = non_null.astype(str).str.strip()

    duplicate_mask = str_values.duplicated(keep=False)
    for idx in str_values.index[duplicate_mask]:
        issues.append(
            _make_issue(
                run_id, dataset_name, detected_at,
                category=IssueCategory.IDENTIFIER.value, issue_type="identifier_duplicate_value",
                severity=Severity.MEDIUM.value, confidence=1.0, column_name=column, row_index=idx,
                current_value=str_values.loc[idx],
                recommended_action="Duplicate identifier value -- review before assuming this is an error; it may be a legitimate repeat (e.g. a return or update).",
                rule_name="identifier_duplicate_value",
            )
        )

    if len(str_values) >= 5:
        signatures = str_values.map(format_signature)
        dominant = signatures.value_counts().idxmax()
        inconsistent_mask = signatures != dominant
        inconsistent_count = int(inconsistent_mask.sum())
        if 0 < inconsistent_count < len(str_values):
            for idx in str_values.index[inconsistent_mask]:
                issues.append(
                    _make_issue(
                        run_id, dataset_name, detected_at,
                        category=IssueCategory.IDENTIFIER.value, issue_type="identifier_inconsistent_format",
                        severity=Severity.LOW.value, confidence=0.6, column_name=column, row_index=idx,
                        current_value=str_values.loc[idx],
                        recommended_action="This value's format differs from the majority of identifiers in this column -- review for consistency.",
                        rule_name="identifier_inconsistent_format",
                    )
                )

    return issues
