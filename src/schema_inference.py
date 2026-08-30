"""Generic logical type inference for an unfamiliar column.

Pandas dtype (int64, object, ...) is a storage detail, not a meaning. This
module infers what a column *represents* -- an email, a date typed as text,
a currency amount, a free-text description -- independently of both the
dtype and the column name.

Column names are used only as a weak, corroborating signal (e.g. a
"price"-like name nudges a decimal column toward "Possible currency", and a
high-uniqueness column named "customer_id" nudges toward "Possible
identifier"). The primary evidence is always the values themselves: parsing
success rates, regex match rates, uniqueness, and format consistency.

Every inference returns a transparent confidence score and the evidence
used to reach it, so the decision can be shown to (and second-guessed by) a
user rather than treated as a black box.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_MAX_UNIQUE_VALUES,
    CATEGORICAL_UNIQUE_RATIO_THRESHOLD,
    CURRENCY_MATCH_THRESHOLD,
    CURRENCY_NAME_HINTS,
    DATE_LIKE_GATE_THRESHOLD,
    DATE_PARSE_SUCCESS_THRESHOLD,
    EMAIL_MATCH_THRESHOLD,
    FREE_TEXT_MIN_AVG_LENGTH,
    FREE_TEXT_MIN_AVG_LENGTH_FOR_UNIQUE_RULE,
    FREE_TEXT_UNIQUE_RATIO_THRESHOLD,
    IDENTIFIER_FORMAT_CONSISTENCY_THRESHOLD,
    IDENTIFIER_MIN_ROWS,
    IDENTIFIER_NAME_HINTS,
    IDENTIFIER_UNIQUE_RATIO_THRESHOLD,
    MIXED_TYPE_LOWER_BOUND,
    MIXED_TYPE_UPPER_BOUND,
    NUMERIC_HIGH_UNIQUENESS_THRESHOLD,
    NUMERIC_LENGTH_CONSISTENCY_THRESHOLD,
    NUMERIC_PHONE_LENGTH_RANGE,
    NUMERIC_SEMANTIC_HIGH_CONFIDENCE,
    NUMERIC_SEMANTIC_MIN_ROWS,
    NUMERIC_SEMANTIC_MODERATE_CONFIDENCE,
    NUMERIC_SEMANTIC_REDUCED_CONFIDENCE,
    NUMERIC_TEXT_THRESHOLD,
    PHONE_MATCH_THRESHOLD,
    PHONE_NAME_HINTS,
    URL_MATCH_THRESHOLD,
)


class LogicalType(str, Enum):
    """Logical (semantic) types a column can be inferred as."""

    EMPTY = "Empty"
    BOOLEAN = "Boolean"
    INTEGER = "Integer"
    DECIMAL = "Decimal"
    NUMERIC_TEXT = "Numeric stored as text"
    DATE = "Date"
    DATETIME = "Datetime"
    DATE_TEXT = "Date stored as text"
    CATEGORICAL = "Categorical"
    FREE_TEXT = "Free text"
    EMAIL = "Email"
    PHONE = "Phone number"
    URL = "URL"
    IDENTIFIER = "Possible identifier"
    CURRENCY = "Possible currency"
    MIXED = "Mixed type"
    UNKNOWN = "Unknown"


@dataclass
class TypeInference:
    """Result of inferring a column's logical type."""

    logical_type: LogicalType
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)


# --- Regex patterns ------------------------------------------------------------

_BOOLEAN_TOKENS = {"true", "false", "yes", "no", "y", "n", "t", "f"}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL_RE = re.compile(r"^(https?://|ftp://|www\.)\S+$", re.IGNORECASE)
_PHONE_PUNCTUATION_RE = re.compile(r"[\s\-().]")

_CURRENCY_RE = re.compile(
    r"^\(?-?\s*(?:[$€£¥₪]|usd|eur|gbp|ils|nis)\s*-?\d[\d,]*\.?\d*\)?$"
    r"|^\(?-?\d[\d,]*\.?\d*\s*(?:[$€£¥₪]|usd|eur|gbp|ils|nis)\)?$",
    re.IGNORECASE,
)

_DATE_LIKE_RE = re.compile(
    r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ t]\d{1,2}:\d{2}(:\d{2})?)?$"
    r"|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}([ t]\d{1,2}:\d{2}(:\d{2})?)?$"
    r"|^\d{8}$"
    r"|^\d{1,2}\s+[a-z]{3,9}\s+\d{2,4}$"
    r"|^[a-z]{3,9}\s+\d{1,2},?\s+\d{2,4}$",
    re.IGNORECASE,
)

_NUMERIC_STRIP_RE = re.compile(r"[$€£¥₪,%\s]")


# --- Small helpers ---------------------------------------------------------------


def normalize_column_name(name: str) -> str:
    """Normalize a column name for display and for weak name-hint matching.

    Lowercases, and collapses any run of non-alphanumeric characters
    (spaces, punctuation) into a single underscore.

    Args:
        name: The original column name.

    Returns:
        A normalized name, e.g. "Customer ID" -> "customer_id". Never empty.
    """
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return normalized.strip("_") or "column"


def _name_matches(normalized_name: str, hints: tuple[str, ...]) -> bool:
    """Check a normalized column name against weak-signal hint words.

    A hint counts as a match if it appears as a whole underscore-separated
    token (e.g. "id" in "customer_id"), or -- for hints longer than two
    characters, to avoid short-substring false positives -- as a suffix of
    the whole name (e.g. "price" in "unitprice").

    Args:
        normalized_name: Output of normalize_column_name().
        hints: Candidate hint words.

    Returns:
        True if any hint matches.
    """
    tokens = normalized_name.split("_")
    if any(token in hints for token in tokens):
        return True
    return any(normalized_name.endswith(hint) for hint in hints if len(hint) > 2)


def _match_rate(values: pd.Series, predicate: Callable[[Any], bool]) -> float:
    """Fraction of values in a Series that satisfy a predicate."""
    if len(values) == 0:
        return 0.0
    return float(values.map(predicate).sum()) / len(values)


def clean_numeric_string(value: str) -> str:
    """Strip currency symbols, thousands separators, and % signs from a value.

    Accounting-style negatives written as "(123.45)" are converted to
    "-123.45".
    """
    cleaned = value.strip()
    is_paren_negative = cleaned.startswith("(") and cleaned.endswith(")")
    if is_paren_negative:
        cleaned = cleaned[1:-1]
    cleaned = _NUMERIC_STRIP_RE.sub("", cleaned)
    if is_paren_negative and cleaned and not cleaned.startswith("-"):
        cleaned = "-" + cleaned
    return cleaned


def is_numeric_string(value: str) -> bool:
    """Whether a string parses as a number after stripping numeric formatting."""
    cleaned = clean_numeric_string(value)
    if cleaned in ("", "-", ".", "-."):
        return False
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _has_at_most_n_decimals(value: float, n: int, tolerance: float = 1e-9) -> bool:
    """Whether a float is (within tolerance) representable with <= n decimals."""
    return abs(round(value, n) - value) < tolerance


def format_signature(value: str) -> tuple[int, bool, bool]:
    """A coarse (length, has_alpha, has_digit) signature used for format consistency."""
    return (len(value), any(c.isalpha() for c in value), any(c.isdigit() for c in value))


def _format_consistency_rate(values: pd.Series) -> float:
    """Fraction of values sharing the single most common format signature."""
    if len(values) == 0:
        return 0.0
    signatures = values.map(format_signature)
    top_count = signatures.value_counts().iloc[0]
    return float(top_count) / len(values)


# --- Per-type match-rate calculators (operate on stripped, non-blank strings) ----


def _boolean_rate(str_values: pd.Series) -> float:
    normalized = str_values.str.lower()
    return _match_rate(normalized, lambda v: v in _BOOLEAN_TOKENS)


def _email_rate(str_values: pd.Series) -> float:
    return _match_rate(str_values, lambda v: bool(_EMAIL_RE.match(v)))


def _url_rate(str_values: pd.Series) -> float:
    return _match_rate(str_values, lambda v: bool(_URL_RE.match(v)))


def _phone_candidate_rate(str_values: pd.Series) -> float:
    def is_phone(v: str) -> bool:
        digits = _PHONE_PUNCTUATION_RE.sub("", v).lstrip("+")
        return digits.isdigit() and 7 <= len(digits) <= 15

    return _match_rate(str_values, is_phone)


def _phone_punctuation_rate(str_values: pd.Series) -> float:
    return _match_rate(str_values, lambda v: bool(_PHONE_PUNCTUATION_RE.search(v)) or v.startswith("+"))


def _currency_rate(str_values: pd.Series) -> float:
    return _match_rate(str_values, lambda v: bool(_CURRENCY_RE.match(v)))


def _date_like_rate(str_values: pd.Series) -> float:
    return _match_rate(str_values, lambda v: bool(_DATE_LIKE_RE.match(v)))


def _numeric_string_rate(str_values: pd.Series) -> float:
    return _match_rate(str_values, is_numeric_string)


# --- Numeric-dtype semantic overrides (Phone / Identifier) -----------------------


def _whole_number_semantic_signals(non_null: pd.Series, normalized_name: str) -> dict[str, Any]:
    """Compute weak, corroborating evidence that a whole-number numeric column is
    really a phone number or an identifier rather than a measurement.

    ``length_consistency`` (how uniform the digit count is) doubles as a rough
    proxy for "does averaging this make sense": genuine measurements naturally
    vary in magnitude (and so in digit count), while phone numbers and
    identifiers are usually formatted to a fixed width.

    Note on leading zeros: by the time a column reaches this function it is
    already a numeric pandas dtype, so any leading zero in the original file
    (e.g. "0501234567") has already been silently dropped by the CSV/Excel
    parser -- there is no way to recover that information here. This is
    surfaced as a caveat in the resulting evidence, not claimed as detected.
    """
    values = non_null.astype("int64")
    str_values = values.astype(str)
    lengths = str_values.str.len()
    length_counts = lengths.value_counts()
    dominant_length = int(length_counts.idxmax())
    length_consistency = float(length_counts.iloc[0]) / len(lengths)
    unique_ratio = float(non_null.nunique()) / len(non_null) if len(non_null) else 0.0
    has_negative = bool((values < 0).any())

    min_len, max_len = NUMERIC_PHONE_LENGTH_RANGE
    phone_length_plausible = not has_negative and min_len <= dominant_length <= max_len

    return {
        "length_consistency": round(length_consistency, 3),
        "dominant_length": dominant_length,
        "unique_ratio": round(unique_ratio, 3),
        "has_negative": has_negative,
        "name_hint_phone": _name_matches(normalized_name, PHONE_NAME_HINTS),
        "name_hint_identifier": _name_matches(normalized_name, IDENTIFIER_NAME_HINTS),
        "phone_length_plausible": phone_length_plausible,
    }


def _classify_whole_number(non_null: pd.Series, normalized_name: str, plain_confidence: float, plain_reason: str) -> TypeInference:
    """Decide between Phone / Possible identifier / Integer for a whole-number column.

    A column-name hint alone never reclassifies the column -- it is
    necessary but not sufficient. It must be corroborated by value-shape
    evidence (consistent digit length and/or high uniqueness) to actually
    switch away from Integer; with a hint but no corroboration, the column
    stays Integer but at reduced confidence (never a forced 100%).

    Args:
        non_null: The column's non-null values (numeric dtype).
        normalized_name: normalize_column_name() output for this column.
        plain_confidence: Confidence to use if no semantic signal applies
            (1.0 for native int64, 0.9 for float-dtype-but-all-whole).
        plain_reason: Evidence "reason" text to use in the plain case.

    Returns:
        A TypeInference for Phone, Possible identifier, or Integer.
    """
    if len(non_null) < NUMERIC_SEMANTIC_MIN_ROWS:
        return TypeInference(LogicalType.INTEGER, plain_confidence, {"reason": plain_reason})

    signals = _whole_number_semantic_signals(non_null, normalized_name)

    if signals["name_hint_phone"]:
        strong_length = signals["phone_length_plausible"]
        strong_uniqueness = signals["unique_ratio"] >= NUMERIC_HIGH_UNIQUENESS_THRESHOLD
        if strong_length or strong_uniqueness:
            confidence = NUMERIC_SEMANTIC_HIGH_CONFIDENCE if (strong_length and strong_uniqueness) else NUMERIC_SEMANTIC_MODERATE_CONFIDENCE
            return TypeInference(
                LogicalType.PHONE, confidence,
                {
                    "reason": "Numeric dtype, but the column name and value shape (digit length/uniqueness) suggest a "
                    "phone number rather than a measurement. Note: any leading zeros in the original file may "
                    "already have been dropped by the numeric dtype -- review the source formatting if unsure.",
                    **signals,
                },
            )
        reduced = round(min(plain_confidence, NUMERIC_SEMANTIC_REDUCED_CONFIDENCE), 3)
        return TypeInference(
            LogicalType.INTEGER, reduced,
            {"reason": plain_reason + " -- column name suggests a phone number, but value shape does not corroborate this.", **signals},
        )

    if signals["name_hint_identifier"]:
        strong_length = signals["length_consistency"] >= NUMERIC_LENGTH_CONSISTENCY_THRESHOLD
        strong_uniqueness = signals["unique_ratio"] >= IDENTIFIER_UNIQUE_RATIO_THRESHOLD
        if strong_length or strong_uniqueness:
            confidence = NUMERIC_SEMANTIC_HIGH_CONFIDENCE if (strong_length and strong_uniqueness) else NUMERIC_SEMANTIC_MODERATE_CONFIDENCE
            return TypeInference(
                LogicalType.IDENTIFIER, confidence,
                {
                    "reason": "Numeric dtype, but the column name and value shape (digit length/uniqueness) suggest "
                    "an identifier rather than a measurement.",
                    **signals,
                },
            )
        reduced = round(min(plain_confidence, NUMERIC_SEMANTIC_REDUCED_CONFIDENCE), 3)
        return TypeInference(
            LogicalType.INTEGER, reduced,
            {"reason": plain_reason + " -- column name suggests an identifier, but value shape does not corroborate this.", **signals},
        )

    return TypeInference(LogicalType.INTEGER, plain_confidence, {"reason": plain_reason})


# --- Main entry point -------------------------------------------------------------


def infer_logical_type(series: pd.Series, column_name: str = "") -> TypeInference:
    """Infer the logical type of a column, independent of its pandas dtype.

    Detection order (most to least specific): native dtype fast paths
    (bool/datetime/int/float, trusted at high confidence) -> pattern-based
    checks on text (email, URL, phone, currency, date-as-text,
    numeric-as-text) -> a "mixed type" gray zone -> uniqueness/format-based
    fallback (possible identifier, categorical, free text, unknown).

    Args:
        series: The column values (any dtype).
        column_name: The original column name, used only as a weak
            corroborating signal (never as the sole basis for a decision).

    Returns:
        A TypeInference with the logical type, a confidence score in
        [0, 1], and an evidence dict explaining the decision.
    """
    non_null = series.dropna()
    if non_null.empty:
        return TypeInference(LogicalType.EMPTY, 1.0, {"reason": "no non-null values"})

    normalized_name = normalize_column_name(column_name)

    if pd.api.types.is_bool_dtype(series):
        return TypeInference(LogicalType.BOOLEAN, 1.0, {"reason": "pandas bool dtype"})

    if pd.api.types.is_datetime64_any_dtype(series):
        has_time_component = bool(
            (non_null.dt.hour != 0).any() or (non_null.dt.minute != 0).any() or (non_null.dt.second != 0).any()
        )
        logical_type = LogicalType.DATETIME if has_time_component else LogicalType.DATE
        return TypeInference(logical_type, 1.0, {"reason": "pandas datetime dtype", "has_time_component": has_time_component})

    if pd.api.types.is_integer_dtype(series):
        if _name_matches(normalized_name, CURRENCY_NAME_HINTS):
            return TypeInference(
                LogicalType.CURRENCY, 0.6, {"reason": "integer dtype with currency-like column name", "column_name": normalized_name}
            )
        return _classify_whole_number(non_null, normalized_name, 1.0, "pandas integer dtype")

    if pd.api.types.is_float_dtype(series):
        with np.errstate(invalid="ignore"):
            all_whole = bool(np.all(np.mod(non_null.to_numpy(dtype=float), 1) == 0))
        if all_whole:
            return _classify_whole_number(non_null, normalized_name, 0.9, "float dtype but all non-null values are whole numbers")
        if _name_matches(normalized_name, CURRENCY_NAME_HINTS):
            two_decimal_rate = _match_rate(non_null, lambda v: _has_at_most_n_decimals(float(v), 2))
            confidence = round(min(0.6 + 0.3 * two_decimal_rate, 0.9), 3)
            return TypeInference(
                LogicalType.CURRENCY,
                confidence,
                {"reason": "decimal dtype with currency-like column name", "two_decimal_rate": round(two_decimal_rate, 3)},
            )
        return TypeInference(LogicalType.DECIMAL, 1.0, {"reason": "pandas float dtype"})

    # --- Object / text dtype: pattern-based detection on stripped strings ---

    str_values = non_null.astype(str).str.strip()
    str_values = str_values[str_values != ""]
    if str_values.empty:
        return TypeInference(LogicalType.EMPTY, 1.0, {"reason": "all values blank after stripping whitespace"})

    n = len(str_values)
    unique_count = int(str_values.nunique())
    unique_ratio = unique_count / n

    rates = {
        "boolean": _boolean_rate(str_values),
        "email": _email_rate(str_values),
        "url": _url_rate(str_values),
        "phone_candidate": _phone_candidate_rate(str_values),
        "currency": _currency_rate(str_values),
        "date_like": _date_like_rate(str_values),
        "numeric": _numeric_string_rate(str_values),
    }

    if rates["boolean"] >= 0.9 and unique_count <= 4:
        return TypeInference(LogicalType.BOOLEAN, round(rates["boolean"], 3), {"rates": rates})

    if rates["email"] >= EMAIL_MATCH_THRESHOLD:
        return TypeInference(LogicalType.EMAIL, round(rates["email"], 3), {"rates": rates})

    if rates["url"] >= URL_MATCH_THRESHOLD:
        return TypeInference(LogicalType.URL, round(rates["url"], 3), {"rates": rates})

    # Currency, date, and numeric patterns are more structurally specific than
    # the loose "digits of plausible phone length" check, so they are tried
    # first -- otherwise dates like "2023-01-01" (digits separated by "-")
    # would be misread as phone numbers before ever reaching the date check.
    if rates["currency"] >= CURRENCY_MATCH_THRESHOLD:
        return TypeInference(LogicalType.CURRENCY, round(rates["currency"], 3), {"rates": rates})

    if rates["date_like"] >= DATE_LIKE_GATE_THRESHOLD:
        # format="mixed" parses each value with its own inferred format,
        # instead of locking the whole column to one format (pandas' default
        # fast path) and turning genuinely valid but differently-formatted
        # dates into NaT.
        parsed = pd.to_datetime(str_values, errors="coerce", format="mixed")
        parse_rate = float(parsed.notna().mean())
        if parse_rate >= DATE_PARSE_SUCCESS_THRESHOLD:
            return TypeInference(LogicalType.DATE_TEXT, round(parse_rate, 3), {"rates": rates, "parse_rate": round(parse_rate, 3)})

    if rates["numeric"] >= NUMERIC_TEXT_THRESHOLD:
        return TypeInference(LogicalType.NUMERIC_TEXT, round(rates["numeric"], 3), {"rates": rates})

    punctuation_rate = _phone_punctuation_rate(str_values)
    name_hint_phone = _name_matches(normalized_name, PHONE_NAME_HINTS)
    if rates["phone_candidate"] >= PHONE_MATCH_THRESHOLD and (punctuation_rate >= 0.3 or name_hint_phone):
        return TypeInference(
            LogicalType.PHONE,
            round(rates["phone_candidate"], 3),
            {"rates": rates, "punctuation_rate": round(punctuation_rate, 3), "name_hint": name_hint_phone},
        )

    max_rate_type, max_rate = max(rates.items(), key=lambda kv: kv[1])
    if MIXED_TYPE_LOWER_BOUND <= max_rate <= MIXED_TYPE_UPPER_BOUND:
        mixed_confidence = round(1 - abs(2 * max_rate - 1), 3)
        return TypeInference(
            LogicalType.MIXED, mixed_confidence, {"rates": rates, "dominant_partial_type": max_rate_type, "dominant_rate": round(max_rate, 3)}
        )

    avg_length = float(str_values.str.len().mean())
    format_consistency = _format_consistency_rate(str_values)
    name_hint_identifier = _name_matches(normalized_name, IDENTIFIER_NAME_HINTS)

    if (
        unique_ratio >= IDENTIFIER_UNIQUE_RATIO_THRESHOLD
        and n >= IDENTIFIER_MIN_ROWS
        and (name_hint_identifier or format_consistency >= IDENTIFIER_FORMAT_CONSISTENCY_THRESHOLD)
    ):
        confidence = 0.5
        confidence += 0.25 if name_hint_identifier else 0.0
        confidence += 0.2 if format_consistency >= IDENTIFIER_FORMAT_CONSISTENCY_THRESHOLD else 0.0
        return TypeInference(
            LogicalType.IDENTIFIER,
            round(min(confidence, 0.95), 3),
            {
                "unique_ratio": round(unique_ratio, 3),
                "format_consistency": round(format_consistency, 3),
                "name_hint": name_hint_identifier,
            },
        )

    has_repeated_values = unique_count < n
    if has_repeated_values and (unique_count <= CATEGORICAL_MAX_UNIQUE_VALUES or unique_ratio <= CATEGORICAL_UNIQUE_RATIO_THRESHOLD):
        confidence = round(min(max(1 - unique_ratio, 0.5), 0.95), 3)
        return TypeInference(
            LogicalType.CATEGORICAL, confidence, {"unique_ratio": round(unique_ratio, 3), "unique_count": unique_count}
        )

    is_long_text = avg_length >= FREE_TEXT_MIN_AVG_LENGTH
    is_unique_text = unique_ratio >= FREE_TEXT_UNIQUE_RATIO_THRESHOLD and avg_length >= FREE_TEXT_MIN_AVG_LENGTH_FOR_UNIQUE_RULE
    if is_long_text or is_unique_text:
        confidence = round(min(0.5 + avg_length / 100, 0.9), 3)
        return TypeInference(
            LogicalType.FREE_TEXT, confidence, {"avg_length": round(avg_length, 2), "unique_ratio": round(unique_ratio, 3)}
        )

    return TypeInference(
        LogicalType.UNKNOWN,
        0.3,
        {"rates": rates, "unique_ratio": round(unique_ratio, 3), "avg_length": round(avg_length, 2), "format_consistency": round(format_consistency, 3)},
    )
