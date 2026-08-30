"""Safe cleaning engine (Phase 4).

Every function here is a *pure* transform: it takes a DataFrame, returns a
brand-new DataFrame (via ``.copy()``) plus the audit entries describing what
changed, and never mutates its input. The caller is always responsible for
holding onto the original -- this module has no notion of "the" dataset,
only "a" DataFrame in, "a" DataFrame out.

Only fixes that are safe to apply without human judgment live here (see
``SAFE_FIX_DEFINITIONS``). Anything requiring a judgment call -- filling
missing values, merging similar categories, removing outliers, resolving
mixed types -- is Phase 3's job to flag, not this module's job to act on.

Row indices are preserved throughout (no ``reset_index``), even across row
removal, so every audit entry's ``row_index`` always refers back to the
original uploaded file's row position, regardless of which other fixes ran
before it in the same cleaning run.

No-data-loss rule: the two type-conversion fixes (date-as-text,
numeric-as-text) only ever convert a column when *100%* of its non-null
values convert successfully. If even one value would fail and become
missing, the whole-column conversion is skipped for that column and
reported back as a note -- never applied partially.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config import load_missing_placeholders
from src.models import AuditLogEntry
from src.schema_inference import LogicalType, clean_numeric_string, infer_logical_type, normalize_column_name

ApplyFn = Callable[[pd.DataFrame, str, str, str], tuple[pd.DataFrame, list[AuditLogEntry], list[str]]]


# --- Shared helpers ----------------------------------------------------------------


def _is_text_column(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def _make_audit_entry(
    run_id: str,
    dataset_name: str,
    timestamp: str,
    *,
    cleaning_action: str,
    rule_name: str,
    reason: str,
    confidence: float,
    row_index: Optional[int] = None,
    column_name: Optional[str] = None,
    original_value: Optional[str] = None,
    new_value: Optional[str] = None,
    user_approved: bool = True,
) -> AuditLogEntry:
    return AuditLogEntry(
        run_id=run_id,
        timestamp=timestamp,
        dataset_name=dataset_name,
        cleaning_action=cleaning_action,
        rule_name=rule_name,
        reason=reason,
        user_approved=user_approved,
        confidence=round(confidence, 3),
        row_index=None if row_index is None else int(row_index),
        column_name=column_name,
        original_value=original_value,
        new_value=new_value,
    )


def _apply_text_transform(
    df: pd.DataFrame, dataset_name: str, run_id: str, timestamp: str, *, action: str, transform: Callable[[str], str], reason: str
) -> tuple[pd.DataFrame, list[AuditLogEntry], list[str]]:
    """Apply a str->str transform to every text column, logging each change."""
    working = df.copy()
    audit: list[AuditLogEntry] = []

    for position, col in enumerate(working.columns):
        series = working.iloc[:, position]
        if not _is_text_column(series):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        original = non_null.astype(str)
        transformed = original.map(transform)
        changed_mask = original != transformed
        if not changed_mask.any():
            continue

        for idx in original.index[changed_mask]:
            audit.append(
                _make_audit_entry(
                    run_id, dataset_name, timestamp,
                    cleaning_action=action, rule_name=action, reason=reason, confidence=1.0,
                    row_index=idx, column_name=str(col),
                    original_value=original.loc[idx], new_value=transformed.loc[idx],
                )
            )
        updated_series = series.copy()
        updated_series.loc[original.index[changed_mask]] = transformed[changed_mask]
        working.isetitem(position, updated_series)

    return working, audit, []


# --- Individual safe fixes ---------------------------------------------------------------


def apply_trim_whitespace(df: pd.DataFrame, dataset_name: str, run_id: str, timestamp: str):
    """Trim leading/trailing whitespace from every text value."""
    return _apply_text_transform(
        df, dataset_name, run_id, timestamp,
        action="trim_whitespace", transform=lambda v: v.strip(),
        reason="Removed leading/trailing whitespace.",
    )


def apply_collapse_internal_spaces(df: pd.DataFrame, dataset_name: str, run_id: str, timestamp: str):
    """Collapse repeated internal whitespace into a single space."""
    return _apply_text_transform(
        df, dataset_name, run_id, timestamp,
        action="collapse_internal_spaces", transform=lambda v: re.sub(r"\s+", " ", v).strip(),
        reason="Collapsed repeated internal spaces into a single space.",
    )


def apply_remove_non_printable(df: pd.DataFrame, dataset_name: str, run_id: str, timestamp: str):
    """Strip non-printable/control characters from text values."""

    def _clean(value: str) -> str:
        return "".join(ch for ch in value if ch.isprintable() or ch in ("\n", "\t"))

    return _apply_text_transform(
        df, dataset_name, run_id, timestamp,
        action="remove_non_printable_characters", transform=_clean,
        reason="Removed non-printable characters.",
    )


def apply_nullify_blank_and_placeholders(df: pd.DataFrame, dataset_name: str, run_id: str, timestamp: str):
    """Convert blank/whitespace-only strings and known placeholders to real missing values."""
    working = df.copy()
    audit: list[AuditLogEntry] = []
    placeholders = load_missing_placeholders()

    for position, col in enumerate(working.columns):
        series = working.iloc[:, position]
        if not _is_text_column(series):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        original = non_null.astype(str)
        stripped = original.str.strip()
        is_blank = stripped == ""
        is_placeholder = ~is_blank & stripped.str.lower().isin(placeholders)
        to_null_mask = is_blank | is_placeholder
        if not to_null_mask.any():
            continue

        for idx in original.index[to_null_mask]:
            reason = (
                "Blank/whitespace-only value converted to a proper missing value."
                if is_blank.loc[idx]
                else "Recognized missing-value placeholder converted to a proper missing value."
            )
            audit.append(
                _make_audit_entry(
                    run_id, dataset_name, timestamp,
                    cleaning_action="nullify_blank_and_placeholder_values", rule_name="nullify_blank_and_placeholder_values",
                    reason=reason, confidence=1.0, row_index=idx, column_name=str(col),
                    original_value=original.loc[idx], new_value=None,
                )
            )
        updated_series = series.copy()
        updated_series.loc[original.index[to_null_mask]] = None
        working.isetitem(position, updated_series)

    return working, audit, []


def apply_normalize_column_names(df: pd.DataFrame, dataset_name: str, run_id: str, timestamp: str):
    """Rename every column to its normalized (snake_case) form."""
    working = df.copy()
    audit: list[AuditLogEntry] = []
    rename_map: dict = {}
    used_names: set = set()

    for col in working.columns:
        base_name = normalize_column_name(str(col))
        final_name = base_name
        suffix = 1
        while final_name in used_names:
            suffix += 1
            final_name = f"{base_name}_{suffix}"
        used_names.add(final_name)
        rename_map[col] = final_name
        if final_name != str(col):
            audit.append(
                _make_audit_entry(
                    run_id, dataset_name, timestamp,
                    cleaning_action="normalize_column_names", rule_name="normalize_column_names",
                    reason="Standardized column name to snake_case.", confidence=1.0,
                    column_name=str(col), original_value=str(col), new_value=final_name,
                )
            )

    working = working.rename(columns=rename_map)
    return working, audit, []


def apply_remove_empty_rows(df: pd.DataFrame, dataset_name: str, run_id: str, timestamp: str):
    """Remove rows that are entirely empty (every column null)."""
    working = df.copy()
    audit: list[AuditLogEntry] = []
    empty_mask = working.isna().all(axis=1)

    for idx in working.index[empty_mask]:
        audit.append(
            _make_audit_entry(
                run_id, dataset_name, timestamp,
                cleaning_action="remove_empty_rows", rule_name="remove_empty_rows",
                reason="Row had no data in any column.", confidence=1.0,
                row_index=idx, original_value="(entire row empty)",
            )
        )

    working = working.loc[~empty_mask]
    return working, audit, []


def apply_remove_empty_columns(df: pd.DataFrame, dataset_name: str, run_id: str, timestamp: str):
    """Remove columns that are entirely empty (every value null)."""
    working = df.copy()
    audit: list[AuditLogEntry] = []
    empty_columns = [col for position, col in enumerate(working.columns) if working.iloc[:, position].isna().all()]

    for col in empty_columns:
        audit.append(
            _make_audit_entry(
                run_id, dataset_name, timestamp,
                cleaning_action="remove_empty_columns", rule_name="remove_empty_columns",
                reason="Column had no data in any row.", confidence=1.0,
                column_name=str(col), original_value="(entire column empty)",
            )
        )

    working = working.drop(columns=empty_columns)
    return working, audit, []


def apply_convert_dates_stored_as_text(df: pd.DataFrame, dataset_name: str, run_id: str, timestamp: str):
    """Convert date-as-text columns to real datetimes, but only with zero data loss.

    A column is converted only if every single non-null value parses
    successfully. If even one value would fail, the column is left
    untouched and a note explains why, per the "never silently lose data"
    requirement.
    """
    working = df.copy()
    audit: list[AuditLogEntry] = []
    notes: list[str] = []

    for position, col in enumerate(working.columns):
        series = working.iloc[:, position]
        if not _is_text_column(series):
            continue
        inference = infer_logical_type(series, str(col))
        if inference.logical_type != LogicalType.DATE_TEXT:
            continue

        non_null = series.dropna()
        original = non_null.astype(str).str.strip()
        parsed = pd.to_datetime(original, errors="coerce", format="mixed")
        failed_count = int(parsed.isna().sum())
        if failed_count:
            notes.append(
                f"'{col}': not converted -- {failed_count} value(s) would fail conversion and become missing."
            )
            continue

        for idx in original.index:
            audit.append(
                _make_audit_entry(
                    run_id, dataset_name, timestamp,
                    cleaning_action="convert_dates_stored_as_text", rule_name="convert_dates_stored_as_text",
                    reason="All non-null values converted successfully to datetime with no data loss.",
                    confidence=inference.confidence, row_index=idx, column_name=str(col),
                    original_value=original.loc[idx], new_value=str(parsed.loc[idx]),
                )
            )

        full_column = pd.Series(pd.NaT, index=working.index, dtype="datetime64[ns]")
        full_column.loc[non_null.index] = parsed.to_numpy()
        working.isetitem(position, full_column)

    return working, audit, notes


def apply_convert_numeric_stored_as_text(df: pd.DataFrame, dataset_name: str, run_id: str, timestamp: str):
    """Convert numeric-as-text columns to real numbers, but only with zero data loss.

    Same no-data-loss rule as date conversion: a column is only converted
    when every non-null value parses successfully.
    """
    working = df.copy()
    audit: list[AuditLogEntry] = []
    notes: list[str] = []

    for position, col in enumerate(working.columns):
        series = working.iloc[:, position]
        if not _is_text_column(series):
            continue
        inference = infer_logical_type(series, str(col))
        if inference.logical_type != LogicalType.NUMERIC_TEXT:
            continue

        non_null = series.dropna()
        original = non_null.astype(str).str.strip()
        numeric = pd.to_numeric(original.map(clean_numeric_string), errors="coerce")
        failed_count = int(numeric.isna().sum())
        if failed_count:
            notes.append(
                f"'{col}': not converted -- {failed_count} value(s) would fail conversion and become missing."
            )
            continue

        for idx in original.index:
            audit.append(
                _make_audit_entry(
                    run_id, dataset_name, timestamp,
                    cleaning_action="convert_numeric_stored_as_text", rule_name="convert_numeric_stored_as_text",
                    reason="All non-null values converted successfully to numeric with no data loss.",
                    confidence=inference.confidence, row_index=idx, column_name=str(col),
                    original_value=original.loc[idx], new_value=str(numeric.loc[idx]),
                )
            )

        full_column = pd.Series(np.nan, index=working.index, dtype="float64")
        full_column.loc[non_null.index] = numeric.to_numpy()
        working.isetitem(position, full_column)

    return working, audit, notes


def apply_remove_exact_duplicate_rows(df: pd.DataFrame, dataset_name: str, run_id: str, timestamp: str):
    """Remove exact duplicate rows (keeping the first occurrence).

    Requires explicit user approval in the UI -- this fix is never applied
    unless the corresponding checkbox is selected.
    """
    working = df.copy()
    audit: list[AuditLogEntry] = []
    duplicate_mask = working.duplicated(keep="first")

    for idx in working.index[duplicate_mask]:
        audit.append(
            _make_audit_entry(
                run_id, dataset_name, timestamp,
                cleaning_action="remove_exact_duplicate_rows", rule_name="remove_exact_duplicate_rows",
                reason="Exact duplicate of an earlier row; removed with explicit user approval.",
                confidence=1.0, row_index=idx, original_value="(duplicate row)",
            )
        )

    working = working.loc[~duplicate_mask]
    return working, audit, []


# --- Fix registry --------------------------------------------------------------------------


@dataclass
class FixDefinition:
    """One selectable safe fix: identity, explanation, and its apply function."""

    fix_id: str
    title: str
    description: str
    apply_fn: ApplyFn
    requires_explicit_approval: bool = False


SAFE_FIX_DEFINITIONS: list[FixDefinition] = [
    FixDefinition(
        "trim_whitespace", "Trim leading/trailing whitespace",
        "Removes spaces at the start or end of text values.", apply_trim_whitespace,
    ),
    FixDefinition(
        "collapse_internal_spaces", "Collapse repeated internal spaces",
        "Replaces runs of multiple spaces inside text values with a single space.", apply_collapse_internal_spaces,
    ),
    FixDefinition(
        "remove_non_printable", "Remove non-printable characters",
        "Strips control/non-printable characters from text values.", apply_remove_non_printable,
    ),
    FixDefinition(
        "nullify_blank_and_placeholders", "Convert blank strings and known placeholders to missing values",
        "Turns empty strings, whitespace-only strings, and recognized placeholders "
        "(e.g. 'N/A', 'unknown') into real missing values -- never fills anything in.",
        apply_nullify_blank_and_placeholders,
    ),
    FixDefinition(
        "normalize_column_names", "Normalize column names",
        "Renames all columns to a consistent snake_case format.", apply_normalize_column_names,
    ),
    FixDefinition(
        "remove_empty_rows", "Remove completely empty rows",
        "Removes rows that have no data in any column.", apply_remove_empty_rows,
    ),
    FixDefinition(
        "remove_empty_columns", "Remove completely empty columns",
        "Removes columns that have no data in any row.", apply_remove_empty_columns,
    ),
    FixDefinition(
        "convert_dates_stored_as_text", "Convert dates stored as text",
        "Converts a column to real dates, but only if every value converts successfully "
        "(no value is ever silently turned into missing data).",
        apply_convert_dates_stored_as_text,
    ),
    FixDefinition(
        "convert_numeric_stored_as_text", "Convert numeric values stored as text",
        "Converts a column to real numbers, but only if every value converts successfully "
        "(no value is ever silently turned into missing data).",
        apply_convert_numeric_stored_as_text,
    ),
    FixDefinition(
        "remove_exact_duplicate_rows", "Remove exact duplicate rows",
        "Permanently removes rows that are an exact copy of an earlier row in the working copy. "
        "Requires your explicit approval -- select this box only if you want duplicates removed.",
        apply_remove_exact_duplicate_rows, requires_explicit_approval=True,
    ),
]


@dataclass
class FixPreview:
    """Read-only preview of what a fix would do, without applying it."""

    fix_id: str
    title: str
    description: str
    requires_explicit_approval: bool
    affected_count: int
    notes: list[str] = field(default_factory=list)


def preview_fixes(df: pd.DataFrame, dataset_name: str) -> list[FixPreview]:
    """Preview every safe fix's effect, applied cumulatively in pipeline order.

    Counts reflect applying all fixes in the order listed -- if you apply
    only a subset, later fixes may find fewer (or more) affected rows than
    shown here (e.g. fewer empty rows if you skip placeholder normalization).

    Args:
        df: The dataset to preview fixes against (never modified).
        dataset_name: Name recorded on audit entries generated internally
            for counting purposes (discarded, not returned).

    Returns:
        One FixPreview per registered safe fix, in application order.
    """
    run_id = "preview"
    timestamp = datetime.now(timezone.utc).isoformat()
    working = df.copy()
    previews: list[FixPreview] = []

    for fix in SAFE_FIX_DEFINITIONS:
        working, audit, notes = fix.apply_fn(working, dataset_name, run_id, timestamp)
        previews.append(
            FixPreview(
                fix_id=fix.fix_id, title=fix.title, description=fix.description,
                requires_explicit_approval=fix.requires_explicit_approval,
                affected_count=len(audit), notes=notes,
            )
        )

    return previews


@dataclass
class CleaningResult:
    """The full result of applying a selected set of safe fixes."""

    run_id: str
    dataset_name: str
    timestamp: str
    cleaned_df: pd.DataFrame
    audit_log: list[AuditLogEntry]
    applied_fix_ids: list[str]
    skipped_ineligible: list[tuple[str, list[str]]]


def apply_selected_fixes(df: pd.DataFrame, selected_fix_ids: list[str], dataset_name: str) -> CleaningResult:
    """Apply only the selected fixes, in fixed pipeline order, to a copy of df.

    Args:
        df: The original (or current working) dataset. Never modified.
        selected_fix_ids: fix_id values the user checked in the UI.
        dataset_name: Recorded on every audit entry.

    Returns:
        A CleaningResult with the cleaned copy, the full audit log, which
        fixes were actually applied, and which selected fixes were skipped
        because they were not eligible (e.g. a date column with values that
        would fail conversion) along with the explanatory notes.
    """
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    working = df.copy()

    all_audit: list[AuditLogEntry] = []
    applied: list[str] = []
    skipped_ineligible: list[tuple[str, list[str]]] = []

    for fix in SAFE_FIX_DEFINITIONS:
        if fix.fix_id not in selected_fix_ids:
            continue
        working, audit, notes = fix.apply_fn(working, dataset_name, run_id, timestamp)
        if audit:
            applied.append(fix.fix_id)
            all_audit.extend(audit)
        if notes:
            skipped_ineligible.append((fix.fix_id, notes))

    return CleaningResult(
        run_id=run_id, dataset_name=dataset_name, timestamp=timestamp,
        cleaned_df=working, audit_log=all_audit, applied_fix_ids=applied,
        skipped_ineligible=skipped_ineligible,
    )


# --- Guided cleaning execution (Phase 7B) -------------------------------------------------
#
# Unlike SAFE_FIX_DEFINITIONS above (a fixed pipeline applied automatically
# once selected), the functions below execute one *user-chosen* remediation
# for one issue group at a time, decided via src.cleaning_strategies. Every
# call here corresponds to an explicit "Apply Cleaning Decision" click --
# nothing here is ever invoked just because a dropdown changed.


def apply_value_replacements(
    df: pd.DataFrame,
    column: str,
    replacements: dict[int, Optional[object]],
    dataset_name: str,
    run_id: str,
    timestamp: str,
    *,
    cleaning_action: str,
    reason: str,
    confidence: float,
) -> tuple[pd.DataFrame, list[AuditLogEntry]]:
    """Set specific rows of one column to specific (possibly per-row) values.

    The single generic primitive behind every guided-cleaning data change:
    missing-value fills, outlier replacement/capping, negative-value
    treatment, and categorical-variant standardization all reduce to "these
    rows in this column become these values" and share this one function,
    so every guided data change is logged identically.

    Args:
        df: Current working copy. Never modified.
        column: Column to update.
        replacements: {row_index: new_value}. A ``None`` value sets that
            cell to a proper missing value (NULL), not the literal text "None".
        dataset_name, run_id, timestamp: Recorded on every audit entry.
        cleaning_action: Slug recorded as both ``cleaning_action`` and
            ``rule_name`` on each audit entry (e.g. "missing_value_replace_median").
        reason: Human-readable explanation recorded on each audit entry.
        confidence: Confidence to record (e.g. the column's inference
            confidence, or 1.0 for a user-provided custom value).

    Returns:
        (new_df, audit_entries) -- one audit entry per row actually changed.
    """
    working = df.copy()
    audit: list[AuditLogEntry] = []
    if not replacements:
        return working, audit

    position = list(working.columns).index(column)
    series = working.iloc[:, position]
    updated_series = series.copy()

    for row_index, new_value in replacements.items():
        original_value = series.loc[row_index] if row_index in series.index else None
        updated_series.loc[row_index] = new_value
        audit.append(
            _make_audit_entry(
                run_id, dataset_name, timestamp,
                cleaning_action=cleaning_action, rule_name=cleaning_action, reason=reason, confidence=confidence,
                row_index=row_index, column_name=column,
                original_value=None if original_value is None or pd.isna(original_value) else str(original_value),
                new_value=None if new_value is None else str(new_value),
            )
        )

    working.isetitem(position, updated_series)
    return working, audit
