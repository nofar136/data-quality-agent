"""Tests for src.type_override -- the human-in-the-loop logical type override."""

from __future__ import annotations

import pandas as pd

from src.profiler import profile_column, profile_dataframe
from src.rule_engine import detect_issues
from src.schema_inference import LogicalType
from src.type_override import (
    USE_DETECTED_LABEL,
    TypeOverrideRecord,
    apply_type_overrides,
    is_low_confidence,
    make_override_record,
)


def test_no_override_leaves_effective_type_equal_to_inferred() -> None:
    profiles = [profile_column(pd.Series([1, 2, 3]), "amount")]
    result = apply_type_overrides(profiles, overrides={})

    assert result[0].effective_logical_type == result[0].logical_type
    assert result[0].type_overridden is False
    assert result[0].user_selected_type is None


def test_user_override_changes_effective_type_only() -> None:
    profiles = [profile_column(pd.Series([str(i) for i in range(30)]), "customer_code")]
    original_logical_type = profiles[0].logical_type

    result = apply_type_overrides(profiles, overrides={"customer_code": "Identifier"})

    overridden = result[0]
    assert overridden.effective_logical_type == LogicalType.IDENTIFIER.value
    assert overridden.type_overridden is True
    assert overridden.user_selected_type == "Identifier"
    # The original inference must be preserved, not replaced.
    assert overridden.logical_type == original_logical_type


def test_resetting_override_via_use_detected_label() -> None:
    profiles = [profile_column(pd.Series([1, 2, 3]), "amount")]
    overridden = apply_type_overrides(profiles, overrides={"amount": "Categorical"})
    assert overridden[0].type_overridden is True

    reset = apply_type_overrides(profiles, overrides={"amount": USE_DETECTED_LABEL})
    assert reset[0].type_overridden is False
    assert reset[0].effective_logical_type == reset[0].logical_type
    assert reset[0].user_selected_type is None


def test_apply_type_overrides_does_not_mutate_input_profiles() -> None:
    profiles = [profile_column(pd.Series([1, 2, 3]), "amount")]
    original_effective_type = profiles[0].effective_logical_type

    apply_type_overrides(profiles, overrides={"amount": "Text"})

    assert profiles[0].effective_logical_type == original_effective_type
    assert profiles[0].type_overridden is False


def test_only_named_columns_are_affected() -> None:
    profiles = profile_dataframe(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
    result = apply_type_overrides(profiles, overrides={"a": "Text"})

    by_name = {p.original_name: p for p in result}
    assert by_name["a"].type_overridden is True
    assert by_name["b"].type_overridden is False


# --- Low-confidence warning ------------------------------------------------------------


def test_is_low_confidence_below_threshold() -> None:
    assert is_low_confidence(0.64, threshold=0.7) is True


def test_is_low_confidence_at_or_above_threshold() -> None:
    assert is_low_confidence(0.70, threshold=0.7) is False
    assert is_low_confidence(0.95, threshold=0.7) is False


# --- Effective type used downstream by issue detection --------------------------------


def test_effective_type_drives_issue_detection() -> None:
    # A column with mixed formats and a neutral name infers as Categorical
    # (no identifier name-hint, inconsistent format) -- but the user knows
    # it's really an identifier. Overriding should make identifier-only
    # checks (e.g. identifier_duplicate_value) start firing where they
    # didn't before.
    values = [f"item-{i}" for i in range(20)] + [f"ITEM_{i:02d}" for i in range(10)] + ["item-0"]
    df = pd.DataFrame({"reference_value": values})
    profiles = profile_dataframe(df)
    assert profiles[0].logical_type == "Categorical"

    baseline_issues = detect_issues(df, profiles, "ds").issues
    baseline_identifier_issues = [i for i in baseline_issues if i.issue_category == "Identifier"]
    assert baseline_identifier_issues == []

    overridden_profiles = apply_type_overrides(profiles, overrides={"reference_value": "Identifier"})
    overridden_issues = detect_issues(df, overridden_profiles, "ds").issues
    overridden_identifier_issues = [i for i in overridden_issues if i.issue_category == "Identifier"]

    assert len(overridden_identifier_issues) > 0


def test_raw_dataframe_is_never_touched_by_override() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    original_copy = df.copy(deep=True)
    profiles = profile_dataframe(df)

    apply_type_overrides(profiles, overrides={"a": "Categorical"})
    detect_issues(df, apply_type_overrides(profiles, overrides={"a": "Categorical"}), "ds")

    pd.testing.assert_frame_equal(df, original_copy)


# --- Audit record -----------------------------------------------------------------------


def test_make_override_record_for_a_changed_type() -> None:
    profile = profile_column(pd.Series([str(i) for i in range(20)]), "customer_code")
    record = make_override_record(profile, "Identifier")

    assert isinstance(record, TypeOverrideRecord)
    assert record.column_name == "customer_code"
    assert record.original_type == profile.logical_type
    assert record.new_type == LogicalType.IDENTIFIER.value
    assert record.user_approved is True
    assert record.timestamp


def test_make_override_record_for_a_confirmation() -> None:
    profile = profile_column(pd.Series([1, 2, 3]), "amount")
    record = make_override_record(profile, USE_DETECTED_LABEL)

    assert record.original_type == profile.logical_type
    assert record.new_type == profile.logical_type
    assert record.user_approved is True


# --- User override from Integer to Phone (real-world HR "Phone" scenario) ------------------


def _numeric_looking_phone_df() -> pd.DataFrame:
    # A fully generic name with no phone/identifier/currency hint at all, so
    # automatic inference has no reason to move off plain Integer -- this
    # isolates the *manual* override path from the automatic-detection path
    # (covered separately in test_schema_inference.py).
    return pd.DataFrame({"raw_value": [501234560 + i for i in range(25)]})


def test_user_can_override_integer_to_phone() -> None:
    df = _numeric_looking_phone_df()
    profiles = profile_dataframe(df)
    assert profiles[0].logical_type == LogicalType.INTEGER.value

    overridden = apply_type_overrides(profiles, overrides={"raw_value": "Phone"})
    assert overridden[0].effective_logical_type == LogicalType.PHONE.value
    assert overridden[0].type_overridden is True
    # The original inference is preserved for comparison, not overwritten.
    assert overridden[0].logical_type == LogicalType.INTEGER.value


def test_effective_phone_type_excludes_numeric_cleaning_checks_downstream() -> None:
    # After overriding to Phone, numeric-only issue types (outliers,
    # infinite values, suspiciously constant column) must never appear --
    # this is what keeps a future Mean/Median/outlier cleaning strategy from
    # ever being offered for a phone-number column.
    df = _numeric_looking_phone_df()
    profiles = profile_dataframe(df)

    overridden_profiles = apply_type_overrides(profiles, overrides={"raw_value": "Phone"})
    issues = detect_issues(df, overridden_profiles, "ds").issues

    numeric_only_issue_types = {"possible_outlier", "infinite_value", "suspiciously_constant_column", "numeric_format_inconsistency"}
    assert not any(i.issue_type in numeric_only_issue_types for i in issues)
    assert not any(i.issue_category == "Numeric" for i in issues)


def test_automatic_phone_detection_also_excludes_numeric_checks() -> None:
    # Even without any manual override, a column that automatically infers
    # as Phone (e.g. named "Phone" -- see test_schema_inference.py) must
    # already be excluded from numeric-only checks by the rule engine.
    df = pd.DataFrame({"Phone": [501234560 + i for i in range(25)]})
    profiles = profile_dataframe(df)
    assert profiles[0].logical_type == LogicalType.PHONE.value
    assert profiles[0].effective_logical_type == LogicalType.PHONE.value

    issues = detect_issues(df, profiles, "ds").issues
    assert not any(i.issue_category == "Numeric" for i in issues)
