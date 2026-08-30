"""Tests for src.cleaning_engine.

Covers each safe fix in isolation, the "never silently lose data" rule for
type conversions, immutability of the input DataFrame, audit log accuracy,
and the guardrails that keep judgment-call fixes (filling missing values,
removing outliers) out of this module entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cleaning_engine import (
    SAFE_FIX_DEFINITIONS,
    apply_convert_dates_stored_as_text,
    apply_convert_numeric_stored_as_text,
    apply_nullify_blank_and_placeholders,
    apply_remove_empty_columns,
    apply_remove_empty_rows,
    apply_remove_exact_duplicate_rows,
    apply_selected_fixes,
    apply_trim_whitespace,
    apply_value_replacements,
    preview_fixes,
)

RUN_ID = "test-run"
TIMESTAMP = "2024-01-01T00:00:00+00:00"


# --- Whitespace cleaning -----------------------------------------------------------


def test_trim_whitespace_cleans_and_logs() -> None:
    df = pd.DataFrame({"name": ["  Alice", "Bob  ", "Carol"]})
    cleaned, audit, notes = apply_trim_whitespace(df, "ds", RUN_ID, TIMESTAMP)

    assert list(cleaned["name"]) == ["Alice", "Bob", "Carol"]
    assert len(audit) == 2
    assert notes == []
    assert {a.row_index for a in audit} == {0, 1}
    assert all(a.cleaning_action == "trim_whitespace" for a in audit)
    assert all(a.confidence == 1.0 for a in audit)


# --- Placeholder / blank normalization -----------------------------------------------


def test_placeholder_and_blank_normalization() -> None:
    df = pd.DataFrame({"status": ["Active", "", "N/A", "  ", "unknown", "Active"]})
    cleaned, audit, notes = apply_nullify_blank_and_placeholders(df, "ds", RUN_ID, TIMESTAMP)

    assert cleaned["status"].isna().sum() == 4
    assert cleaned.loc[0, "status"] == "Active"
    assert len(audit) == 4
    assert all(a.new_value is None for a in audit)
    assert notes == []


def test_placeholder_normalization_never_fills_values() -> None:
    # A genuinely missing value must remain missing, not be replaced with anything.
    df = pd.DataFrame({"status": ["Active", None, "N/A"]})
    cleaned, _, _ = apply_nullify_blank_and_placeholders(df, "ds", RUN_ID, TIMESTAMP)

    assert cleaned["status"].isna().sum() == 2
    assert cleaned.loc[1, "status"] is None or pd.isna(cleaned.loc[1, "status"])


# --- Empty-row / empty-column removal --------------------------------------------------


def test_empty_row_removal_preserves_original_row_labels() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
    cleaned, audit, _ = apply_remove_empty_rows(df, "ds", RUN_ID, TIMESTAMP)

    assert list(cleaned.index) == [0, 2]
    assert len(audit) == 1
    assert audit[0].row_index == 1
    assert audit[0].cleaning_action == "remove_empty_rows"


def test_empty_column_removal() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [None, None]})
    cleaned, audit, _ = apply_remove_empty_columns(df, "ds", RUN_ID, TIMESTAMP)

    assert list(cleaned.columns) == ["a"]
    assert len(audit) == 1
    assert audit[0].column_name == "b"


# --- Exact duplicate removal (requires explicit approval) -------------------------------


def test_exact_duplicate_removal_standalone() -> None:
    df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    cleaned, audit, _ = apply_remove_exact_duplicate_rows(df, "ds", RUN_ID, TIMESTAMP)

    assert list(cleaned.index) == [0, 1]
    assert len(audit) == 1
    assert audit[0].row_index == 2


def test_duplicate_removal_not_applied_unless_selected() -> None:
    df = pd.DataFrame({"a": [1, 2, 1]})
    result = apply_selected_fixes(df, ["trim_whitespace"], "ds")

    assert result.cleaned_df.shape[0] == 3  # duplicate still present
    assert "remove_exact_duplicate_rows" not in result.applied_fix_ids


def test_duplicate_removal_applied_when_selected() -> None:
    df = pd.DataFrame({"a": [1, 2, 1]})
    result = apply_selected_fixes(df, ["remove_exact_duplicate_rows"], "ds")

    assert result.cleaned_df.shape[0] == 2
    assert "remove_exact_duplicate_rows" in result.applied_fix_ids
    fix_def = next(f for f in SAFE_FIX_DEFINITIONS if f.fix_id == "remove_exact_duplicate_rows")
    assert fix_def.requires_explicit_approval is True


# --- Safe date conversion / failed conversion -----------------------------------------


def test_safe_date_conversion_when_all_values_convert() -> None:
    dates = [f"2023-01-{i:02d}" for i in range(1, 10)]
    df = pd.DataFrame({"event_date": dates})
    cleaned, audit, notes = apply_convert_dates_stored_as_text(df, "ds", RUN_ID, TIMESTAMP)

    assert pd.api.types.is_datetime64_any_dtype(cleaned["event_date"])
    assert len(audit) == len(dates)
    assert notes == []


def test_date_conversion_skipped_when_any_value_fails() -> None:
    dates = [f"2023-01-{i:02d}" for i in range(1, 10)] + ["not-a-date"]
    df = pd.DataFrame({"event_date": dates})
    cleaned, audit, notes = apply_convert_dates_stored_as_text(df, "ds", RUN_ID, TIMESTAMP)

    # Column must be left completely untouched -- no partial conversion.
    assert cleaned["event_date"].tolist() == dates
    assert audit == []
    assert len(notes) == 1
    assert "not converted" in notes[0]


def test_date_conversion_preserves_missing_values() -> None:
    dates = [f"2023-01-{i:02d}" for i in range(1, 10)] + [None]
    df = pd.DataFrame({"event_date": dates})
    cleaned, audit, notes = apply_convert_dates_stored_as_text(df, "ds", RUN_ID, TIMESTAMP)

    assert notes == []
    assert pd.isna(cleaned["event_date"].iloc[-1])
    assert len(audit) == 9  # only non-null values are logged


# --- Numeric-text conversion --------------------------------------------------------------


def test_numeric_text_conversion_when_all_values_convert() -> None:
    values = [str(i) for i in range(15)]
    df = pd.DataFrame({"amount": values})
    cleaned, audit, notes = apply_convert_numeric_stored_as_text(df, "ds", RUN_ID, TIMESTAMP)

    assert pd.api.types.is_numeric_dtype(cleaned["amount"])
    assert cleaned["amount"].tolist() == [float(i) for i in range(15)]
    assert len(audit) == 15
    assert notes == []


def test_numeric_text_conversion_skipped_on_partial_failure() -> None:
    values = [str(i) for i in range(9)] + ["abc"]
    df = pd.DataFrame({"amount": values})
    cleaned, audit, notes = apply_convert_numeric_stored_as_text(df, "ds", RUN_ID, TIMESTAMP)

    assert cleaned["amount"].tolist() == values
    assert audit == []
    assert len(notes) == 1


# --- Original dataset immutability -------------------------------------------------------


@pytest.mark.parametrize(
    "apply_fn",
    [
        apply_trim_whitespace,
        apply_nullify_blank_and_placeholders,
        apply_remove_empty_rows,
        apply_remove_empty_columns,
        apply_remove_exact_duplicate_rows,
    ],
)
def test_original_dataframe_is_never_mutated(apply_fn) -> None:
    df = pd.DataFrame({"a": ["  x", "", "y", "  x"], "b": [None, None, None, None]})
    original_copy = df.copy(deep=True)

    apply_fn(df, "ds", RUN_ID, TIMESTAMP)

    pd.testing.assert_frame_equal(df, original_copy)


def test_apply_selected_fixes_never_mutates_input() -> None:
    df = pd.DataFrame({"a": ["  x ", "", "N/A", "  x "], "b": [1, 2, 2, 1]})
    original_copy = df.copy(deep=True)

    apply_selected_fixes(df, [f.fix_id for f in SAFE_FIX_DEFINITIONS], "ds")

    pd.testing.assert_frame_equal(df, original_copy)


# --- Audit log accuracy --------------------------------------------------------------------


def test_audit_log_entries_have_required_fields() -> None:
    df = pd.DataFrame({"name": ["  Alice", "Bob"]})
    result = apply_selected_fixes(df, ["trim_whitespace"], "my_dataset.csv")

    assert len(result.audit_log) == 1
    entry = result.audit_log[0]
    assert entry.run_id == result.run_id
    assert entry.dataset_name == "my_dataset.csv"
    assert entry.timestamp == result.timestamp
    assert entry.row_index == 0
    assert entry.column_name == "name"
    assert entry.original_value == "  Alice"
    assert entry.new_value == "Alice"
    assert entry.cleaning_action == "trim_whitespace"
    assert entry.rule_name == "trim_whitespace"
    assert entry.reason
    assert entry.user_approved is True
    assert entry.confidence == 1.0


# --- Reset functionality (no fixes selected == unchanged data) --------------------------


def test_no_fixes_selected_leaves_data_unchanged() -> None:
    df = pd.DataFrame({"name": ["  Alice", "Bob "], "amount": [1, 2]})
    result = apply_selected_fixes(df, [], "ds")

    pd.testing.assert_frame_equal(result.cleaned_df, df)
    assert result.audit_log == []
    assert result.applied_fix_ids == []


# --- Missing values are never filled; outliers are never removed ------------------------


def test_missing_values_never_filled_by_any_safe_fix() -> None:
    # Missing values land on different rows so neither row is fully empty
    # (a fully empty row is legitimately removed by remove_empty_rows,
    # which is not the same thing as filling a missing value).
    df = pd.DataFrame({"amount": [1.0, np.nan, 3.0, 4.0], "notes": ["a", "b", None, "d"]})
    result = apply_selected_fixes(df, [f.fix_id for f in SAFE_FIX_DEFINITIONS], "ds")

    assert result.cleaned_df["amount"].isna().sum() == 1
    assert result.cleaned_df["notes"].isna().sum() == 1


def test_outliers_are_never_removed_by_any_safe_fix() -> None:
    # A unique id column ensures no row is an exact duplicate of another,
    # isolating the outlier-handling behavior from duplicate-row removal.
    df = pd.DataFrame(
        {
            "id": range(9),
            "value": [10, 11, 9, 10, 12, 11, 9, 10, 1000],
        }
    )
    result = apply_selected_fixes(df, [f.fix_id for f in SAFE_FIX_DEFINITIONS], "ds")

    assert result.cleaned_df.shape[0] == 9
    assert 1000 in result.cleaned_df["value"].tolist()


def test_negative_values_are_never_changed() -> None:
    df = pd.DataFrame({"delta": [-5, -3, -1, 2, 4]})
    result = apply_selected_fixes(df, [f.fix_id for f in SAFE_FIX_DEFINITIONS], "ds")

    assert sorted(result.cleaned_df["delta"].tolist()) == sorted([-5, -3, -1, 2, 4])


# --- Fix registry / preview consistency ---------------------------------------------------


def test_fix_registry_has_expected_fix_ids() -> None:
    expected = {
        "trim_whitespace", "collapse_internal_spaces", "remove_non_printable",
        "nullify_blank_and_placeholders", "normalize_column_names", "remove_empty_rows",
        "remove_empty_columns", "convert_dates_stored_as_text", "convert_numeric_stored_as_text",
        "remove_exact_duplicate_rows",
    }
    assert {f.fix_id for f in SAFE_FIX_DEFINITIONS} == expected


# --- apply_value_replacements (Phase 7B guided-cleaning execution primitive) ----------------


def test_apply_value_replacements_sets_specific_rows() -> None:
    df = pd.DataFrame({"salary": [1000.0, None, 3000.0, None]})
    new_df, audit = apply_value_replacements(
        df, "salary", {1: 615.0, 3: 615.0}, "ds", RUN_ID, TIMESTAMP,
        cleaning_action="missing_value_replace_median", reason="User chose median.", confidence=1.0,
    )
    assert new_df["salary"].tolist() == [1000.0, 615.0, 3000.0, 615.0]
    assert len(audit) == 2
    assert {a.row_index for a in audit} == {1, 3}
    assert all(a.new_value == "615.0" for a in audit)
    assert all(a.original_value is None for a in audit)


def test_apply_value_replacements_supports_per_row_values() -> None:
    df = pd.DataFrame({"email": ["a@example.com", None, None]})
    new_df, audit = apply_value_replacements(
        df, "email", {1: "b@example.com", 2: "c@example.com"}, "ds", RUN_ID, TIMESTAMP,
        cleaning_action="missing_value_custom_per_row", reason="User-provided per-row values.", confidence=1.0,
    )
    assert new_df["email"].tolist() == ["a@example.com", "b@example.com", "c@example.com"]
    assert len(audit) == 2


def test_apply_value_replacements_can_set_null() -> None:
    df = pd.DataFrame({"value": [10, 1000, 30]})
    new_df, audit = apply_value_replacements(
        df, "value", {1: None}, "ds", RUN_ID, TIMESTAMP,
        cleaning_action="outlier_set_null", reason="User chose to null the outlier.", confidence=1.0,
    )
    assert pd.isna(new_df.loc[1, "value"])
    assert audit[0].new_value is None
    assert audit[0].original_value == "1000"


def test_apply_value_replacements_does_not_mutate_original() -> None:
    df = pd.DataFrame({"value": [10, None, 30]})
    original_copy = df.copy(deep=True)
    apply_value_replacements(df, "value", {1: 20}, "ds", RUN_ID, TIMESTAMP, cleaning_action="x", reason="x", confidence=1.0)
    pd.testing.assert_frame_equal(df, original_copy)


def test_apply_value_replacements_empty_dict_makes_no_changes() -> None:
    df = pd.DataFrame({"value": [10, None, 30]})
    new_df, audit = apply_value_replacements(df, "value", {}, "ds", RUN_ID, TIMESTAMP, cleaning_action="x", reason="x", confidence=1.0)
    pd.testing.assert_frame_equal(new_df, df)
    assert audit == []


def test_preview_fixes_matches_apply_selected_fixes_counts() -> None:
    df = pd.DataFrame({"name": ["  Alice", "Bob  ", "  Alice"], "id": [1, 2, 1]})
    previews = preview_fixes(df, "ds")
    trim_preview = next(p for p in previews if p.fix_id == "trim_whitespace")

    result = apply_selected_fixes(df, ["trim_whitespace"], "ds")
    assert trim_preview.affected_count == len(result.audit_log)
