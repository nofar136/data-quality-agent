"""Tests for src.rule_engine -- the orchestrator that selects checks per column type."""

from __future__ import annotations

import pandas as pd

from src.profiler import profile_dataframe
from src.rule_engine import detect_issues


def test_detect_issues_returns_accurate_summary_counts() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 4],
            "notes": ["ok", "ok", "ok", None],
        }
    )
    profiles = profile_dataframe(df)
    result = detect_issues(df, profiles, "sample.csv")

    assert result.summary.total_issues == len(result.issues)
    assert sum(result.summary.by_severity.values()) == result.summary.total_issues
    assert sum(result.summary.by_category.values()) == result.summary.total_issues
    assert sum(result.summary.by_type.values()) == result.summary.total_issues


def test_every_issue_has_required_fields_populated() -> None:
    df = pd.DataFrame({"amount": ["10", "20", "abc"], "date": ["2023-01-01", "2023-02-01", "2023-03-01"]})
    profiles = profile_dataframe(df)
    result = detect_issues(df, profiles, "sample.csv")

    assert result.issues, "expected at least one issue to be detected"
    for issue in result.issues:
        assert issue.run_id == result.run_id
        assert issue.dataset_name == "sample.csv"
        assert issue.issue_id
        assert issue.issue_category
        assert issue.issue_type
        assert issue.severity in ("Low", "Medium", "High", "Critical")
        assert 0.0 <= issue.confidence <= 1.0
        assert issue.recommended_action
        assert issue.rule_name
        assert issue.detected_at == result.detected_at


def test_checks_are_selected_by_logical_type_not_column_name() -> None:
    # Column named "phone_number" but the values are plainly free text --
    # detection must follow the inferred type, not the name.
    df = pd.DataFrame(
        {
            "phone_number": [
                "This is a fairly long free text note about the customer's preferences.",
                "Another long note describing a completely unrelated topic in detail.",
                "Yet another descriptive sentence that has nothing to do with phones.",
                "A fourth note, again just prose, not a phone number in sight here.",
            ]
        }
    )
    profiles = profile_dataframe(df)
    result = detect_issues(df, profiles, "sample.csv")

    # No identifier-only checks should have fired for a free-text column.
    assert not any(i.issue_category == "Identifier" for i in result.issues)


def test_clean_dataset_has_no_critical_issues() -> None:
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:04d}" for i in range(50)],
            "email": [f"user{i}@example.com" for i in range(50)],
            "amount": [round(9.99 + i, 2) for i in range(50)],
            "category": (["Electronics", "Groceries", "Clothing", "Books", "Toys"] * 10),
            "signup_date": pd.date_range("2023-01-01", periods=50, freq="D"),
        }
    )
    profiles = profile_dataframe(df)
    result = detect_issues(df, profiles, "clean_dataset.csv")

    critical_issues = [i for i in result.issues if i.severity == "Critical"]
    assert critical_issues == []


def test_empty_dataframe_produces_no_issues() -> None:
    df = pd.DataFrame()
    profiles = profile_dataframe(df)
    result = detect_issues(df, profiles, "empty.csv")

    assert result.issues == []
    assert result.summary.total_issues == 0
