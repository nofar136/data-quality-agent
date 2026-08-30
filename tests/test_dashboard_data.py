"""Tests for src.dashboard_data -- pure data prep for the Data Quality Dashboard."""

from __future__ import annotations

import pandas as pd
import pytest

from src.cleaning_engine import SAFE_FIX_DEFINITIONS, apply_selected_fixes
from src.dashboard_data import (
    cleaning_impact_summary,
    compute_kpis,
    issues_by_category_comparison,
    issues_by_severity_comparison,
    issues_total_comparison,
    most_problematic_columns,
    quality_component_comparison,
    unresolved_issues_summary,
)
from src.profiler import profile_dataframe
from src.rule_engine import detect_issues
from src.scoring import calculate_quality_score


def _detect(df: pd.DataFrame, name: str = "ds"):
    result = detect_issues(df, profile_dataframe(df), name)
    return result.issues, result.summary


def _dirty_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"ID-{i:03d}" for i in range(30)] + ["ID-000"],
            "amount": [str(i) for i in range(29)] + ["abc", "10"],
            "notes": ["  hi  ", "ok"] + ["fine"] * 27 + [None, "N/A"],
        }
    )


def _clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"ID-{i:03d}" for i in range(30)],
            "amount": list(range(30)),
            "notes": ["fine"] * 30,
        }
    )


# --- Dashboard before cleaning / after cleaning -----------------------------------------


def test_dashboard_before_cleaning_only() -> None:
    df = _dirty_df()
    issues, summary = _detect(df)
    score = calculate_quality_score(df, issues, "original")

    kpis = compute_kpis(score, summary)

    assert kpis.original_score == score.overall_score
    assert kpis.cleaned_score is None
    assert kpis.score_improvement is None
    assert kpis.total_issues_after is None
    assert kpis.issues_resolved is None
    assert kpis.total_issues_before == summary.total_issues


def test_dashboard_after_cleaning() -> None:
    df = _dirty_df()
    issues, summary = _detect(df)
    score = calculate_quality_score(df, issues, "original")

    cleaning_result = apply_selected_fixes(df, [f.fix_id for f in SAFE_FIX_DEFINITIONS], "ds")
    cleaned_issues, cleaned_summary = _detect(cleaning_result.cleaned_df)
    cleaned_score = calculate_quality_score(cleaning_result.cleaned_df, cleaned_issues, "cleaned")

    kpis = compute_kpis(score, summary, cleaned_score, cleaned_summary, len(cleaning_result.audit_log))

    assert kpis.cleaned_score == cleaned_score.overall_score
    assert kpis.total_issues_after == cleaned_summary.total_issues
    assert kpis.total_changes_applied == len(cleaning_result.audit_log)


# --- Issues resolved / score improvement calculations ------------------------------------


def test_issues_resolved_calculation() -> None:
    df = _dirty_df()
    issues, summary = _detect(df)
    score = calculate_quality_score(df, issues, "original")

    cleaning_result = apply_selected_fixes(df, [f.fix_id for f in SAFE_FIX_DEFINITIONS], "ds")
    cleaned_issues, cleaned_summary = _detect(cleaning_result.cleaned_df)
    cleaned_score = calculate_quality_score(cleaning_result.cleaned_df, cleaned_issues, "cleaned")

    kpis = compute_kpis(score, summary, cleaned_score, cleaned_summary, len(cleaning_result.audit_log))

    assert kpis.issues_resolved == summary.total_issues - cleaned_summary.total_issues


def test_score_improvement_calculation() -> None:
    df = _dirty_df()
    issues, summary = _detect(df)
    score = calculate_quality_score(df, issues, "original")

    cleaned_score = calculate_quality_score(_clean_df(), [], "cleaned")
    cleaned_summary_issues, cleaned_summary = _detect(_clean_df())

    kpis = compute_kpis(score, summary, cleaned_score, cleaned_summary, total_changes_applied=5)

    assert kpis.score_improvement == pytest.approx(round(cleaned_score.overall_score - score.overall_score, 2))


# --- Category / severity before-after aggregation -----------------------------------------


def test_issue_category_before_after_aggregation() -> None:
    df = _dirty_df()
    issues, summary = _detect(df)
    cleaning_result = apply_selected_fixes(df, [f.fix_id for f in SAFE_FIX_DEFINITIONS], "ds")
    _, cleaned_summary = _detect(cleaning_result.cleaned_df)

    result = issues_by_category_comparison(summary, cleaned_summary)

    assert list(result.columns) == ["category", "before", "after"]
    assert set(result["category"]) == set(summary.by_category) | set(cleaned_summary.by_category)
    for _, row in result.iterrows():
        assert row["before"] == summary.by_category.get(row["category"], 0)
        assert row["after"] == cleaned_summary.by_category.get(row["category"], 0)


def test_issue_severity_before_after_aggregation() -> None:
    df = _dirty_df()
    issues, summary = _detect(df)
    cleaning_result = apply_selected_fixes(df, [f.fix_id for f in SAFE_FIX_DEFINITIONS], "ds")
    _, cleaned_summary = _detect(cleaning_result.cleaned_df)

    result = issues_by_severity_comparison(summary, cleaned_summary)

    assert list(result.columns) == ["severity", "before", "after"]
    # Must respect Critical -> High -> Medium -> Low ordering.
    severities = list(result["severity"])
    assert severities == sorted(severities, key=lambda s: ["Critical", "High", "Medium", "Low"].index(s))


def test_issues_total_comparison_before_only() -> None:
    _, summary = _detect(_dirty_df())
    result = issues_total_comparison(summary)
    assert list(result["state"]) == ["Before"]
    assert result.loc[0, "issue_count"] == summary.total_issues


# --- Problematic column ranking -----------------------------------------------------------


def test_problematic_column_ranking_orders_by_issue_count() -> None:
    issues, _ = _detect(_dirty_df())
    result = most_problematic_columns(issues, top_n=5)

    assert list(result.columns) == ["column_name", "before"]
    assert (result["before"].diff().dropna() <= 0).all()  # non-increasing


def test_problematic_column_ranking_with_cleaned_data() -> None:
    df = _dirty_df()
    issues, _ = _detect(df)
    cleaning_result = apply_selected_fixes(df, [f.fix_id for f in SAFE_FIX_DEFINITIONS], "ds")
    cleaned_issues, _ = _detect(cleaning_result.cleaned_df)

    result = most_problematic_columns(issues, cleaned_issues, top_n=5)
    assert list(result.columns) == ["column_name", "before", "after"]


# --- Quality component comparison -----------------------------------------------------------


def test_quality_component_comparison_before_and_after() -> None:
    df = _dirty_df()
    issues, _ = _detect(df)
    score = calculate_quality_score(df, issues, "original")

    cleaned_df = _clean_df()
    cleaned_issues, _ = _detect(cleaned_df)
    cleaned_score = calculate_quality_score(cleaned_df, cleaned_issues, "cleaned")

    result = quality_component_comparison(score, cleaned_score)
    assert list(result.columns) == ["component", "before", "after"]
    assert len(result) == 5


# --- Cleaning impact -----------------------------------------------------------------------


def test_cleaning_impact_summary_matches_audit_log() -> None:
    df = _dirty_df()
    cleaning_result = apply_selected_fixes(df, [f.fix_id for f in SAFE_FIX_DEFINITIONS], "ds")

    result = cleaning_impact_summary(cleaning_result.audit_log)
    assert result["affected_values"].sum() == len(cleaning_result.audit_log)


def test_cleaning_impact_summary_empty_without_cleaning() -> None:
    result = cleaning_impact_summary(None)
    assert result.empty
    assert list(result.columns) == ["cleaning_action", "affected_values", "affected_rows", "affected_columns"]


# --- Unresolved issues ----------------------------------------------------------------------


def test_unresolved_issues_summary_excludes_safe_to_fix() -> None:
    issues, _ = _detect(_dirty_df())
    result = unresolved_issues_summary(issues)

    assert not result.empty
    unresolved_types = set(result["issue_type"])
    safe_types = {i.issue_type for i in issues if i.safe_to_auto_fix}
    assert not (unresolved_types & safe_types - (unresolved_types - safe_types))
    # No issue type present in the result should be exclusively safe-to-fix.
    for issue_type in unresolved_types:
        assert any(i.issue_type == issue_type and not i.safe_to_auto_fix for i in issues)


# --- Zero-issue / no-cleaned-data behavior --------------------------------------------------


def test_dashboard_behavior_with_zero_issues() -> None:
    df = _clean_df()
    issues, summary = _detect(df)

    assert issues_by_category_comparison(summary).empty
    assert issues_by_severity_comparison(summary).empty
    assert most_problematic_columns(issues).empty
    assert unresolved_issues_summary(issues).empty


def test_dashboard_behavior_without_cleaned_data() -> None:
    issues, summary = _detect(_dirty_df())
    score = calculate_quality_score(_dirty_df(), issues, "original")

    kpis = compute_kpis(score, summary)
    assert kpis.cleaned_score is None

    category_df = issues_by_category_comparison(summary)
    assert "after" not in category_df.columns

    severity_df = issues_by_severity_comparison(summary)
    assert "after" not in severity_df.columns

    columns_df = most_problematic_columns(issues)
    assert "after" not in columns_df.columns

    component_df = quality_component_comparison(score)
    assert "after" not in component_df.columns

    assert cleaning_impact_summary(None).empty
