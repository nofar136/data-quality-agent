"""Tests for src.schema_inference.

Each logical type gets at least one representative column so the detection
rules are verified independently of any specific dataset or column naming
convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.schema_inference import LogicalType, infer_logical_type, normalize_column_name


def _logical_type(series: pd.Series, column_name: str = "value") -> LogicalType:
    return infer_logical_type(series, column_name).logical_type


# --- normalize_column_name ------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Customer ID", "customer_id"),
        ("  Order-Date  ", "order_date"),
        ("Price($)", "price"),
        ("already_normal", "already_normal"),
        ("", "column"),
    ],
)
def test_normalize_column_name(raw: str, expected: str) -> None:
    assert normalize_column_name(raw) == expected


# --- Empty -----------------------------------------------------------------------


def test_empty_column_all_null() -> None:
    series = pd.Series([None, np.nan, None])
    assert _logical_type(series) == LogicalType.EMPTY


def test_empty_column_all_blank_strings() -> None:
    series = pd.Series(["", "   ", None, ""])
    assert _logical_type(series) == LogicalType.EMPTY


# --- Boolean -----------------------------------------------------------------------


def test_boolean_native_dtype() -> None:
    series = pd.Series([True, False, True, False])
    result = infer_logical_type(series, "is_active")
    assert result.logical_type == LogicalType.BOOLEAN
    assert result.confidence == 1.0


def test_boolean_text_values() -> None:
    series = pd.Series(["Yes", "No", "Yes", "No", "yes"])
    assert _logical_type(series, "has_subscription") == LogicalType.BOOLEAN


# --- Integer / Decimal ------------------------------------------------------------


def test_integer_native_dtype() -> None:
    series = pd.Series([1, 2, 3, 4, 5])
    result = infer_logical_type(series, "quantity")
    assert result.logical_type == LogicalType.INTEGER
    assert result.confidence == 1.0


def test_decimal_native_dtype() -> None:
    series = pd.Series([1.1, 2.25, 3.75, 4.0])
    result = infer_logical_type(series, "score")
    assert result.logical_type == LogicalType.DECIMAL


def test_float_dtype_with_only_whole_numbers_is_integer() -> None:
    # Missing values force an int column to float64 in pandas.
    series = pd.Series([1.0, 2.0, np.nan, 4.0])
    assert _logical_type(series, "quantity") == LogicalType.INTEGER


# --- Numeric stored as text --------------------------------------------------------


def test_numeric_stored_as_text() -> None:
    series = pd.Series(["1", "2", "3", "4", "1,000"], dtype=object)
    result = infer_logical_type(series, "amount_text")
    assert result.logical_type == LogicalType.NUMERIC_TEXT
    assert result.confidence >= 0.9


# --- Date / Datetime / Date stored as text -----------------------------------------


def test_date_native_dtype() -> None:
    series = pd.to_datetime(pd.Series(["2023-01-01", "2023-02-15", "2023-03-20"]))
    assert _logical_type(series, "order_date") == LogicalType.DATE


def test_datetime_native_dtype() -> None:
    series = pd.to_datetime(pd.Series(["2023-01-01 10:30:00", "2023-01-02 11:45:00"]))
    assert _logical_type(series, "created_at") == LogicalType.DATETIME


def test_date_stored_as_text() -> None:
    series = pd.Series(["2023-01-01", "2023-02-15", "2023-03-20", "2023-04-10"], dtype=object)
    result = infer_logical_type(series, "signup_date")
    assert result.logical_type == LogicalType.DATE_TEXT
    assert result.confidence >= 0.85


# --- Categorical / Free text --------------------------------------------------------


def test_categorical_low_cardinality() -> None:
    series = pd.Series((["red", "blue", "green"] * 10))
    assert _logical_type(series, "color") == LogicalType.CATEGORICAL


def test_free_text_long_descriptions() -> None:
    series = pd.Series(
        [
            "This product exceeded my expectations in every possible way imaginable.",
            "Delivery was slow but the item quality made up for the long wait time.",
            "I would not recommend this to a friend, the packaging was badly damaged.",
            "Absolutely fantastic value for money, will definitely purchase again soon.",
        ]
    )
    result = infer_logical_type(series, "review_text")
    assert result.logical_type == LogicalType.FREE_TEXT


# --- Email / Phone / URL ------------------------------------------------------------


def test_email() -> None:
    series = pd.Series(["a@example.com", "b.smith@test.org", "c_d@sub.site.net"])
    result = infer_logical_type(series, "contact")
    assert result.logical_type == LogicalType.EMAIL
    assert result.confidence == 1.0


def test_phone_number_with_punctuation() -> None:
    series = pd.Series(["+1-555-123-4567", "(555) 234-5678", "555-345-6789"])
    result = infer_logical_type(series, "contact_number")
    assert result.logical_type == LogicalType.PHONE


def test_url() -> None:
    series = pd.Series(["https://example.com", "http://test.org/page", "https://www.site.net/a/b"])
    result = infer_logical_type(series, "website")
    assert result.logical_type == LogicalType.URL
    assert result.confidence == 1.0


# --- Possible identifier -------------------------------------------------------------


def test_possible_identifier_with_name_hint() -> None:
    series = pd.Series([f"CUST-{i:05d}" for i in range(50)])
    result = infer_logical_type(series, "customer_id")
    assert result.logical_type == LogicalType.IDENTIFIER
    assert result.confidence >= 0.5


def test_high_cardinality_without_hint_is_not_forced_to_identifier() -> None:
    # High uniqueness alone must not be enough -- no name hint, no consistent format.
    rng = np.random.default_rng(42)
    values = [f"note {i} - {rng.integers(0, 999999)} misc" for i in range(40)]
    result = infer_logical_type(pd.Series(values), "comments")
    assert result.logical_type != LogicalType.IDENTIFIER


# --- Possible currency -----------------------------------------------------------------


def test_currency_with_symbol() -> None:
    series = pd.Series(["$10.00", "$25.50", "$5.99", "$100.00"])
    result = infer_logical_type(series, "transaction_amount")
    assert result.logical_type == LogicalType.CURRENCY


def test_currency_from_decimal_dtype_with_name_hint() -> None:
    series = pd.Series([9.99, 19.99, 5.50, 100.00])
    result = infer_logical_type(series, "unit_price")
    assert result.logical_type == LogicalType.CURRENCY


def test_decimal_dtype_without_name_hint_stays_decimal() -> None:
    series = pd.Series([9.99, 19.99, 5.50, 100.00])
    result = infer_logical_type(series, "measurement")
    assert result.logical_type == LogicalType.DECIMAL


# --- Mixed type -------------------------------------------------------------------------


def test_mixed_type_partial_numeric() -> None:
    series = pd.Series(["123", "abc", "456", "xyz", "789", "qrs"])
    result = infer_logical_type(series, "field")
    assert result.logical_type == LogicalType.MIXED


# --- Unknown ----------------------------------------------------------------------------


def test_unknown_fallback() -> None:
    # Highly unique, short, format-inconsistent tokens with no name hint or pattern match.
    values = ["a1", "!!", "?9", "z2q", "w0", "e", "t8x", "u", "i6", "o1p", "p", "kk9"]
    result = infer_logical_type(pd.Series(values), "misc_field")
    assert result.logical_type == LogicalType.UNKNOWN
    assert result.confidence < 0.5


# --- Evidence is always present ----------------------------------------------------------


def test_evidence_dict_is_never_empty() -> None:
    series = pd.Series(["1", "2", "3"])
    result = infer_logical_type(series, "x")
    assert isinstance(result.evidence, dict)
    assert len(result.evidence) > 0


# --- Numeric-dtype semantic override: Phone / Identifier (Phase 7A correction) ------------


def test_numeric_looking_phone_column_is_not_forced_to_integer() -> None:
    # Digits-only phone numbers parse as int64 by pandas -- must not be
    # treated as a numeric measurement despite the numeric dtype.
    phones = pd.Series([501234560 + i for i in range(25)])
    assert phones.dtype.kind == "i"

    result = infer_logical_type(phones, "Phone")
    assert result.logical_type == LogicalType.PHONE
    assert result.confidence < 1.0
    assert "leading zero" in result.evidence["reason"].lower()


@pytest.mark.parametrize("column_name", ["Phone Number", "Mobile"])
def test_other_phone_like_column_names_are_recognized(column_name: str) -> None:
    phones = pd.Series([501234560 + i for i in range(25)])
    result = infer_logical_type(phones, column_name)
    assert result.logical_type == LogicalType.PHONE


@pytest.mark.parametrize("column_name", ["Employee ID", "Customer ID"])
def test_numeric_looking_identifier_columns_are_not_forced_to_integer(column_name: str) -> None:
    employee_ids = pd.Series(range(10001, 10001 + 25))
    assert employee_ids.dtype.kind == "i"

    result = infer_logical_type(employee_ids, column_name)
    assert result.logical_type == LogicalType.IDENTIFIER
    assert result.confidence < 1.0


def test_normal_numeric_measurement_columns_remain_numeric() -> None:
    # A generic name with no phone/identifier hint must stay a plain,
    # fully-confident Integer -- column name is a necessary, not sufficient,
    # trigger, so an unrelated name never causes a false reclassification.
    quantities = pd.Series(range(1, 26))
    result = infer_logical_type(quantities, "Quantity")
    assert result.logical_type == LogicalType.INTEGER
    assert result.confidence == 1.0

    # Even a column whose values happen to have consistent digit length and
    # high uniqueness (e.g. years) must not be reclassified without a
    # corroborating name hint -- name is necessary, value-shape alone is not
    # sufficient either.
    years = pd.Series(range(2000, 2025))
    result_years = infer_logical_type(years, "Year")
    assert result_years.logical_type == LogicalType.INTEGER
    assert result_years.confidence == 1.0


def test_name_hint_alone_without_corroboration_reduces_confidence_but_does_not_reclassify() -> None:
    # "id" name hint present, but the values are low-cardinality, short,
    # inconsistent-length numbers -- not identifier-shaped. The hint alone
    # must not force a reclassification (never the sole factor), only a
    # reduced confidence.
    values = pd.Series([1, 22, 333, 4444, 5, 66, 777, 8888, 9, 10, 11, 12, 13, 14, 15, 1, 22, 333, 4444, 5, 66, 777, 8888, 9, 10])
    result = infer_logical_type(values, "some_id_ish_field")
    assert result.logical_type == LogicalType.INTEGER
    assert result.confidence < 1.0


def test_small_sample_size_is_never_reclassified() -> None:
    # Below the minimum row count, length/uniqueness signals are unreliable,
    # so even a perfectly name-hinted, uniform-looking column stays plain.
    phones = pd.Series([501234561, 501234562, 501234563])
    result = infer_logical_type(phones, "Phone")
    assert result.logical_type == LogicalType.INTEGER
    assert result.confidence == 1.0


def test_currency_name_hint_still_takes_priority_over_phone_identifier_checks() -> None:
    prices = pd.Series(range(10, 35))
    result = infer_logical_type(prices, "Price")
    assert result.logical_type == LogicalType.CURRENCY
