"""Tests for src.profiler."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.profiler import profile_column, profile_dataframe
from src.schema_inference import LogicalType


def test_profile_dataframe_returns_one_profile_per_column_in_order() -> None:
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "amount": [1.5, 2.5, 3.5]})
    profiles = profile_dataframe(df)

    assert [p.original_name for p in profiles] == ["id", "name", "amount"]


def test_profile_column_basic_completeness_stats() -> None:
    series = pd.Series([1, 2, 2, None, 4])
    profile = profile_column(series, "value")

    assert profile.non_null_count == 4
    assert profile.missing_count == 1
    assert profile.missing_pct == 20.0
    assert profile.unique_count == 3
    assert profile.duplicate_value_count == 1


def test_profile_column_normalized_name() -> None:
    profile = profile_column(pd.Series([1, 2, 3]), "Order  ID")
    assert profile.normalized_name == "order_id"


def test_profile_column_numeric_stats() -> None:
    series = pd.Series([10.5, 20.0, 30.25, 40.0, 50.0])
    profile = profile_column(series, "score")

    assert profile.logical_type == LogicalType.DECIMAL.value
    assert profile.min_value == 10.5
    assert profile.max_value == 50.0
    assert profile.mean == 30.15
    assert profile.median == 30.25
    assert profile.std is not None


def test_profile_column_numeric_text_is_parsed_for_stats() -> None:
    series = pd.Series(["$10.00", "$20.00", "$30.00", "$40.00"])
    profile = profile_column(series, "transaction_amount")

    assert profile.logical_type == LogicalType.CURRENCY.value
    assert profile.min_value == 10.0
    assert profile.max_value == 40.0
    assert profile.mean == 25.0


def test_profile_column_outlier_count_detects_extreme_value() -> None:
    series = pd.Series([10, 11, 9, 10, 12, 11, 9, 10, 1000])
    profile = profile_column(series, "value")

    assert profile.outlier_count is not None
    assert profile.outlier_count >= 1


def test_profile_column_date_stats() -> None:
    series = pd.Series(["2023-01-01", "2023-06-15", "2023-12-31"])
    profile = profile_column(series, "event_date")

    assert profile.logical_type == LogicalType.DATE_TEXT.value
    assert profile.min_date is not None
    assert profile.max_date is not None
    assert "2023-01-01" in profile.min_date
    assert "2023-12-31" in profile.max_date


def test_profile_column_text_length_stats() -> None:
    series = pd.Series(["short", "a bit longer text", "the longest piece of text here"])
    profile = profile_column(series, "description")

    assert profile.min_text_length == float(len("short"))
    assert profile.max_text_length == float(len("the longest piece of text here"))
    assert profile.avg_text_length is not None


def test_profile_column_example_values_are_capped() -> None:
    series = pd.Series([f"value_{i}" for i in range(100)])
    profile = profile_column(series, "notes")

    assert len(profile.example_values) <= 5


def test_profile_column_empty_column_has_no_extra_stats() -> None:
    series = pd.Series([None, None, np.nan])
    profile = profile_column(series, "unused")

    assert profile.logical_type == LogicalType.EMPTY.value
    assert profile.min_value is None
    assert profile.min_date is None
    assert profile.avg_text_length is None


def test_profile_dataframe_end_to_end_mixed_dataset() -> None:
    df = pd.DataFrame(
        {
            "Transaction ID": [f"TXN-{i:04d}" for i in range(20)],
            "Customer Email": [f"user{i}@example.com" for i in range(20)],
            "Amount": [round(9.99 + i, 2) for i in range(20)],
            "Category": (["Electronics", "Groceries", "Clothing", "Books"] * 5),
            "Purchase Date": [f"2023-01-{(i % 28) + 1:02d}" for i in range(20)],
            "Notes": [f"Customer left a fairly detailed comment about order number {i}." for i in range(20)],
        }
    )

    profiles = profile_dataframe(df)
    by_name = {p.original_name: p for p in profiles}

    assert by_name["Transaction ID"].logical_type == LogicalType.IDENTIFIER.value
    assert by_name["Customer Email"].logical_type == LogicalType.EMAIL.value
    assert by_name["Amount"].logical_type == LogicalType.CURRENCY.value
    assert by_name["Category"].logical_type == LogicalType.CATEGORICAL.value
    assert by_name["Purchase Date"].logical_type == LogicalType.DATE_TEXT.value
    assert by_name["Notes"].logical_type == LogicalType.FREE_TEXT.value
