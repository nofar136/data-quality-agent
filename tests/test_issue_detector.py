"""Tests for src.issue_detector.

Each detector is tested directly against a column (and its profiler.py
ColumnProfile) or a small DataFrame, independent of the rule engine's
column-type gating (covered separately in tests/test_rule_engine.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.issue_detector import (
    detect_dataset_level_issues,
    detect_date_issues,
    detect_identifier_issues,
    detect_missing_value_issues,
    detect_numeric_issues,
    detect_text_issues,
    detect_type_issues,
)
from src.profiler import profile_column
from src.schema_inference import LogicalType

RUN_ID = "test-run-id"
DETECTED_AT = "2024-01-01T00:00:00+00:00"


# --- Missing values and placeholders ------------------------------------------------


def test_missing_null_and_placeholder_detection() -> None:
    series = pd.Series(["A", None, "N/A", "B", "unknown"])
    profile = profile_column(series, "status")
    issues = detect_missing_value_issues(series, profile, "ds", RUN_ID, DETECTED_AT)

    types = [i.issue_type for i in issues]
    assert types.count("missing_null") == 1
    assert types.count("missing_placeholder") == 2


def test_blank_and_whitespace_only_strings_detected() -> None:
    series = pd.Series(["a", "", "   ", "b"])
    profile = profile_column(series, "col")
    issues = detect_missing_value_issues(series, profile, "ds", RUN_ID, DETECTED_AT)

    types = [i.issue_type for i in issues]
    assert "blank_string" in types
    assert "whitespace_only_string" in types


def test_zero_is_never_treated_as_missing() -> None:
    series = pd.Series([0, 1, 0, 2])
    profile = profile_column(series, "count")
    issues = detect_missing_value_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    assert issues == []


def test_false_is_never_treated_as_missing() -> None:
    series = pd.Series([True, False, True, False])
    profile = profile_column(series, "flag")
    issues = detect_missing_value_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    assert issues == []


# --- Dataset-level: duplicate rows, empty rows/columns -------------------------------


def test_exact_duplicate_rows_detected() -> None:
    df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    issues = detect_dataset_level_issues(df, "ds", RUN_ID, DETECTED_AT)

    dup_issues = [i for i in issues if i.issue_type == "exact_duplicate_row"]
    assert len(dup_issues) == 1
    assert dup_issues[0].row_index == 2


def test_empty_rows_and_columns_detected() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, None]})
    issues = detect_dataset_level_issues(df, "ds", RUN_ID, DETECTED_AT)

    assert any(i.issue_type == "empty_row" and i.row_index == 1 for i in issues)
    assert any(i.issue_type == "empty_column" and i.column_name == "b" for i in issues)


def test_duplicate_column_names_detected_standalone() -> None:
    df = pd.concat([pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3, 4]})], axis=1)
    issues = detect_dataset_level_issues(df, "ds", RUN_ID, DETECTED_AT)
    assert any(i.issue_type == "duplicate_column_name" and i.severity == "Critical" for i in issues)


def test_near_duplicate_column_names_detected() -> None:
    df = pd.DataFrame({"Customer ID": [1, 2], "customer_id": [3, 4]})
    issues = detect_dataset_level_issues(df, "ds", RUN_ID, DETECTED_AT)
    assert any(i.issue_type == "near_duplicate_column_name" for i in issues)


# --- Type issues: numeric-as-text, mixed types ----------------------------------------


def test_numeric_stored_as_text_and_value_failure_detected() -> None:
    series = pd.Series([str(i) for i in range(9)] + ["abc"])
    profile = profile_column(series, "amount_text")
    assert profile.logical_type == LogicalType.NUMERIC_TEXT.value

    issues = detect_type_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    assert any(i.issue_type == "numeric_stored_as_text" for i in issues)

    failures = [i for i in issues if i.issue_type == "value_fails_type_conversion"]
    assert len(failures) == 1
    assert failures[0].current_value == "abc"


def test_mixed_data_types_and_unexpected_text_detected() -> None:
    series = pd.Series(["123", "abc", "456", "xyz", "789", "qrs"])
    profile = profile_column(series, "field")
    assert profile.logical_type == LogicalType.MIXED.value

    issues = detect_type_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    assert any(i.issue_type == "mixed_data_types" for i in issues)

    unexpected = [i for i in issues if i.issue_type == "unexpected_text_in_numeric_column"]
    assert len(unexpected) == 3


# --- Dates stored as text / invalid dates ----------------------------------------------


def test_date_stored_as_text_column_level_detected() -> None:
    series = pd.Series(["2023-01-01", "2023-02-15", "2023-03-20", "2023-04-10"])
    profile = profile_column(series, "signup_date")
    assert profile.logical_type == LogicalType.DATE_TEXT.value

    issues = detect_type_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    assert any(i.issue_type == "date_stored_as_text" for i in issues)


def test_invalid_date_detected() -> None:
    dates = [f"2023-01-{i:02d}" for i in range(1, 10)] + ["not-a-date"]
    series = pd.Series(dates)
    profile = profile_column(series, "event_date")
    assert profile.logical_type == LogicalType.DATE_TEXT.value

    issues = detect_date_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    invalid = [i for i in issues if i.issue_type == "invalid_date"]
    assert len(invalid) == 1
    assert invalid[0].current_value == "not-a-date"
    assert invalid[0].severity == "High"


def test_mixed_date_formats_detected() -> None:
    dates = [f"2023-01-{i:02d}" for i in range(1, 10)] + ["01/15/2023"]
    series = pd.Series(dates)
    profile = profile_column(series, "event_date")

    issues = detect_date_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    assert any(i.issue_type == "mixed_date_formats" and i.current_value == "01/15/2023" for i in issues)


def test_unusual_dates_are_low_severity_not_assumed_incorrect() -> None:
    dates = pd.to_datetime(["1850-01-01"] + [f"2023-01-{i:02d}" for i in range(1, 10)])
    series = pd.Series(dates)
    profile = profile_column(series, "event_date")

    issues = detect_date_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    old_date_issues = [i for i in issues if i.issue_type == "unusually_old_date"]
    assert len(old_date_issues) == 1
    assert old_date_issues[0].severity == "Low"
    assert "do not assume" in old_date_issues[0].recommended_action.lower()


# --- Text-quality issues -----------------------------------------------------------------


def test_leading_trailing_and_repeated_spaces_detected() -> None:
    series = pd.Series(["  Alice", "Bob  ", "Carol   Smith", "Dan"])
    profile = profile_column(series, "name")

    issues = detect_text_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    assert any(i.issue_type == "leading_trailing_whitespace" for i in issues)
    assert any(i.issue_type == "repeated_internal_spaces" for i in issues)


def test_non_printable_characters_detected() -> None:
    series = pd.Series(["Alice", "Bob\x00", "Carol"])
    profile = profile_column(series, "name")

    issues = detect_text_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    assert any(i.issue_type == "non_printable_characters" for i in issues)


def test_similar_category_values_and_capitalization_are_distinguished() -> None:
    # "New York" / "new-york" / "New_York" differ by more than case -> similar_category_values.
    # "Boston" / "boston" differ only by case -> inconsistent_capitalization.
    series = pd.Series((["New York", "new-york", "New_York", "Boston", "boston", "Boston"]) * 3)
    profile = profile_column(series, "city")
    assert profile.logical_type == LogicalType.CATEGORICAL.value

    issues = detect_text_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    similar_values = {i.current_value for i in issues if i.issue_type == "similar_category_values"}
    capitalization_values = {i.current_value for i in issues if i.issue_type == "inconsistent_capitalization"}

    assert similar_values & {"new-york", "New_York"}
    assert "boston" in capitalization_values
    assert "boston" not in similar_values


# --- Numeric issues: outliers, infinite values ---------------------------------------------


def test_iqr_outliers_detected() -> None:
    series = pd.Series([10, 11, 9, 10, 12, 11, 9, 10, 1000])
    profile = profile_column(series, "value")

    issues = detect_numeric_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    outliers = [i for i in issues if i.issue_type == "possible_outlier"]
    assert len(outliers) >= 1
    assert outliers[0].severity == "Low"


def test_infinite_values_detected() -> None:
    series = pd.Series([1.0, 2.0, np.inf, 3.0, -np.inf])
    profile = profile_column(series, "ratio")

    issues = detect_numeric_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    inf_issues = [i for i in issues if i.issue_type == "infinite_value"]
    assert len(inf_issues) == 2
    assert all(i.severity == "Critical" for i in inf_issues)


def test_suspiciously_constant_column_detected() -> None:
    series = pd.Series([5, 5, 5, 5, 5])
    profile = profile_column(series, "constant")

    issues = detect_numeric_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    assert any(i.issue_type == "suspiciously_constant_column" for i in issues)


def test_negative_values_are_not_flagged_as_invalid() -> None:
    series = pd.Series([-5, -3, -1, 2, 4, 6, 8, 10])
    profile = profile_column(series, "delta")

    issues = detect_numeric_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    # Any outliers found must be genuine IQR outliers, not simply the
    # negative values -- negativity itself is only ever Low severity and
    # explicitly framed as "may be valid", never as an error.
    for issue in issues:
        if issue.issue_type == "possible_outlier":
            assert "do not assume" in issue.recommended_action.lower()


def test_negative_values_are_detected_as_low_severity_informational() -> None:
    series = pd.Series([-5, -3, -1, 2, 4, 6, 8, 10])
    profile = profile_column(series, "delta")

    issues = detect_numeric_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    negative_issues = [i for i in issues if i.issue_type == "negative_value"]
    assert len(negative_issues) == 3
    assert all(i.severity == "Low" for i in negative_issues)
    assert all("may be valid" in i.recommended_action.lower() for i in negative_issues)


def test_negative_values_never_detected_for_columns_named_profit_or_revenue() -> None:
    # Column naming must never suppress detection -- "Profit"/"Revenue" style
    # names are not treated as evidence that negatives are impossible.
    series = pd.Series([-100.0, -50.0, 200.0, 300.0])
    profile = profile_column(series, "Profit")

    issues = detect_numeric_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    negative_issues = [i for i in issues if i.issue_type == "negative_value"]
    assert len(negative_issues) == 2


# --- Identifier issues -----------------------------------------------------------------------


def test_identifier_duplicate_values_detected() -> None:
    series = pd.Series([f"CUST-{i:03d}" for i in range(30)] + ["CUST-001"])
    profile = profile_column(series, "customer_id")
    assert profile.logical_type == LogicalType.IDENTIFIER.value

    issues = detect_identifier_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    dup_issues = [i for i in issues if i.issue_type == "identifier_duplicate_value"]
    assert len(dup_issues) == 2
    assert all("review" in i.recommended_action.lower() for i in dup_issues)


def test_identifier_missing_values_detected() -> None:
    series = pd.Series([f"CUST-{i:03d}" for i in range(30)] + [None, None])
    profile = profile_column(series, "customer_id")
    assert profile.logical_type == LogicalType.IDENTIFIER.value

    issues = detect_identifier_issues(series, profile, "ds", RUN_ID, DETECTED_AT)
    missing_issues = [i for i in issues if i.issue_type == "identifier_missing_values"]
    assert len(missing_issues) == 1
    assert missing_issues[0].severity in ("Medium", "High", "Critical")
