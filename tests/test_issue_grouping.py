"""Tests for src.issue_grouping."""

from __future__ import annotations

import pandas as pd

from src.issue_grouping import GUIDED_REVIEW_ISSUE_TYPES, build_issue_groups
from src.profiler import profile_dataframe
from src.rule_engine import detect_issues


def _detect(df: pd.DataFrame):
    profiles = profile_dataframe(df)
    profiles_by_name = {p.original_name: p for p in profiles}
    issues = detect_issues(df, profiles, "ds").issues
    return issues, profiles_by_name


def test_groups_are_keyed_by_column_and_issue_type() -> None:
    df = pd.DataFrame({"salary": [1000.0, None, None, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0]})
    issues, profiles_by_name = _detect(df)

    groups = build_issue_groups(issues, profiles_by_name)
    missing_groups = [g for g in groups if g.issue_type == "missing_null"]
    assert len(missing_groups) == 1
    assert missing_groups[0].column_name == "salary"
    assert missing_groups[0].affected_count == 2
    assert set(missing_groups[0].row_indices) == {1, 2}


def test_only_guided_review_eligible_issue_types_are_grouped() -> None:
    # column_name_whitespace / leading_trailing_whitespace etc. are handled
    # by the existing automatic safe fixes, not the guided workflow.
    df = pd.DataFrame({" Name ": ["  Alice", "Bob"]})
    issues, profiles_by_name = _detect(df)

    groups = build_issue_groups(issues, profiles_by_name)
    assert all(g.issue_type in GUIDED_REVIEW_ISSUE_TYPES for g in groups)
    assert groups == []  # none of this dataset's issues are guided-review types


def test_groups_carry_effective_logical_type_and_confidence() -> None:
    # value_fails_type_conversion (Type category) is not a guided-review
    # issue type, but the missing values in the same column are.
    df = pd.DataFrame({"amount": [str(i) for i in range(28)] + ["abc", None]})
    issues, profiles_by_name = _detect(df)

    groups = build_issue_groups(issues, profiles_by_name)
    assert groups  # at least one guided group should exist
    for group in groups:
        profile = profiles_by_name[group.column_name]
        assert group.effective_logical_type == profile.effective_logical_type
        assert group.inference_confidence == profile.confidence


def test_groups_are_sorted_by_severity_then_affected_count() -> None:
    df = pd.DataFrame(
        {
            "id": range(20),
            "salary": [1000.0] * 15 + [None] * 5,  # High severity (missing ~25%)
            "value": [10, 11, 9, 10, 12, 11, 9, 10, 1000, 10, 11, 9, 10, 12, 11, 9, 10, 11, 9, 10],  # Low severity outlier
        }
    )
    issues, profiles_by_name = _detect(df)
    groups = build_issue_groups(issues, profiles_by_name)

    severities = [g.severity for g in groups]
    severity_rank = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
    ranks = [severity_rank[s] for s in severities]
    assert ranks == sorted(ranks, reverse=True)


def test_dataset_level_issues_without_a_column_are_never_grouped() -> None:
    df = pd.DataFrame({"a": [1, None], "b": [None, None]})  # b is fully empty (dataset-level issue)
    issues, profiles_by_name = _detect(df)

    groups = build_issue_groups(issues, profiles_by_name)
    assert all(g.column_name for g in groups)
