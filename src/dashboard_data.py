"""Pure data-prep functions for the Data Quality Dashboard (Phase 7A).

Deliberately separate from app.py: every function here takes already-computed
domain objects (DetectionSummary, Issue lists, QualityScore, AuditLogEntry
lists) and returns a plain DataFrame or dataclass -- no Streamlit, no Plotly,
nothing that requires a running app to test. app.py wraps these results in
chart widgets; this module only decides *what the numbers are*.

Every "before vs after" function accepts an optional cleaned-side argument
and, when it's None, returns a "before"-only result rather than raising or
guessing -- the dashboard is fully usable before any cleaning has happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.models import AuditLogEntry, DetectionSummary, Issue
from src.scoring import QualityScore

SEVERITY_ORDER: tuple[str, ...] = ("Critical", "High", "Medium", "Low")


@dataclass
class DashboardKPIs:
    """Headline numbers for the dashboard's KPI cards.

    Every "after cleaning" field is None when no cleaning has been applied
    yet, rather than 0 or a guess -- the UI shows "N/A" for None, not a
    misleading zero.
    """

    original_score: float
    cleaned_score: Optional[float]
    score_improvement: Optional[float]
    total_issues_before: int
    total_issues_after: Optional[int]
    issues_resolved: Optional[int]
    total_changes_applied: int


def compute_kpis(
    original_score: QualityScore,
    original_summary: DetectionSummary,
    cleaned_score: Optional[QualityScore] = None,
    cleaned_summary: Optional[DetectionSummary] = None,
    total_changes_applied: int = 0,
) -> DashboardKPIs:
    """Compute the dashboard's KPI card values.

    Args:
        original_score: Quality score for the original (uploaded) dataset.
        original_summary: Issue detection summary for the original dataset.
        cleaned_score: Quality score for the cleaned dataset, if cleaning was applied.
        cleaned_summary: Issue detection summary for the cleaned dataset, if applied.
        total_changes_applied: Number of audit log entries (0 if no cleaning).

    Returns:
        A DashboardKPIs with "after"/"improvement" fields set only when both
        cleaned_score and cleaned_summary are provided.
    """
    has_cleaned = cleaned_score is not None and cleaned_summary is not None
    return DashboardKPIs(
        original_score=original_score.overall_score,
        cleaned_score=cleaned_score.overall_score if has_cleaned else None,
        score_improvement=round(cleaned_score.overall_score - original_score.overall_score, 2) if has_cleaned else None,
        total_issues_before=original_summary.total_issues,
        total_issues_after=cleaned_summary.total_issues if has_cleaned else None,
        issues_resolved=(original_summary.total_issues - cleaned_summary.total_issues) if has_cleaned else None,
        total_changes_applied=total_changes_applied,
    )


def issues_total_comparison(original_summary: DetectionSummary, cleaned_summary: Optional[DetectionSummary] = None) -> pd.DataFrame:
    """Total issue count before (and after, if available) cleaning.

    Returns:
        A DataFrame with columns ["state", "issue_count"] -- one row
        ("Before") if no cleaned summary, two rows ("Before"/"After") otherwise.
    """
    rows = [{"state": "Before", "issue_count": original_summary.total_issues}]
    if cleaned_summary is not None:
        rows.append({"state": "After", "issue_count": cleaned_summary.total_issues})
    return pd.DataFrame(rows)


def issues_by_category_comparison(original_summary: DetectionSummary, cleaned_summary: Optional[DetectionSummary] = None) -> pd.DataFrame:
    """Issue counts by category, before vs after.

    Returns:
        A DataFrame with columns ["category", "before"] or
        ["category", "before", "after"] when cleaned_summary is given.
        Empty (but correctly columned) if there are no issues at all.
    """
    categories = sorted(set(original_summary.by_category) | (set(cleaned_summary.by_category) if cleaned_summary else set()))
    columns = ["category", "before"] if cleaned_summary is None else ["category", "before", "after"]
    if not categories:
        return pd.DataFrame(columns=columns)

    rows = []
    for category in categories:
        row = {"category": category, "before": original_summary.by_category.get(category, 0)}
        if cleaned_summary is not None:
            row["after"] = cleaned_summary.by_category.get(category, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def issues_by_severity_comparison(original_summary: DetectionSummary, cleaned_summary: Optional[DetectionSummary] = None) -> pd.DataFrame:
    """Issue counts by severity, before vs after, in Critical->Low order.

    Returns:
        A DataFrame with columns ["severity", "before"] or
        ["severity", "before", "after"]. Empty if there are no issues at all.
    """
    present = set(original_summary.by_severity) | (set(cleaned_summary.by_severity) if cleaned_summary else set())
    ordered = [s for s in SEVERITY_ORDER if s in present]
    columns = ["severity", "before"] if cleaned_summary is None else ["severity", "before", "after"]
    if not ordered:
        return pd.DataFrame(columns=columns)

    rows = []
    for severity in ordered:
        row = {"severity": severity, "before": original_summary.by_severity.get(severity, 0)}
        if cleaned_summary is not None:
            row["after"] = cleaned_summary.by_severity.get(severity, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def most_problematic_columns(original_issues: list[Issue], cleaned_issues: Optional[list[Issue]] = None, top_n: int = 10) -> pd.DataFrame:
    """Rank columns by issue count, most first.

    Args:
        original_issues: Issues detected in the original dataset.
        cleaned_issues: Issues detected in the cleaned dataset, if available.
        top_n: Maximum number of columns to return.

    Returns:
        A DataFrame with columns ["column_name", "before"] or
        ["column_name", "before", "after"], ranked by "before" descending.
        Empty if no issue references a column (e.g. only dataset-level issues,
        or zero issues).
    """
    before_counts: dict[str, int] = {}
    for issue in original_issues:
        if issue.column_name:
            before_counts[issue.column_name] = before_counts.get(issue.column_name, 0) + 1

    columns = ["column_name", "before"] if cleaned_issues is None else ["column_name", "before", "after"]
    if not before_counts:
        return pd.DataFrame(columns=columns)

    top_columns = sorted(before_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    after_counts: Optional[dict[str, int]] = None
    if cleaned_issues is not None:
        after_counts = {}
        for issue in cleaned_issues:
            if issue.column_name:
                after_counts[issue.column_name] = after_counts.get(issue.column_name, 0) + 1

    rows = []
    for column_name, before_count in top_columns:
        row = {"column_name": column_name, "before": before_count}
        if after_counts is not None:
            row["after"] = after_counts.get(column_name, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def quality_component_comparison(original_score: QualityScore, cleaned_score: Optional[QualityScore] = None) -> pd.DataFrame:
    """The five component scores, before vs after.

    Returns:
        A DataFrame with columns ["component", "before"] or
        ["component", "before", "after"], one row per component, in the
        original score's component order.
    """
    cleaned_by_name = {c.component_name: c.score for c in cleaned_score.components} if cleaned_score is not None else {}
    rows = []
    for component in original_score.components:
        row = {"component": component.component_name, "before": component.score}
        if cleaned_score is not None:
            row["after"] = cleaned_by_name.get(component.component_name)
        rows.append(row)
    return pd.DataFrame(rows)


def cleaning_impact_summary(audit_log: Optional[list[AuditLogEntry]]) -> pd.DataFrame:
    """Per-fix-type rollup of what cleaning actually changed.

    Args:
        audit_log: Entries from a CleaningResult, or None/empty if no
            cleaning has been applied yet.

    Returns:
        A DataFrame with columns ["cleaning_action", "affected_values",
        "affected_rows", "affected_columns"], sorted by affected_values
        descending. Empty if no cleaning has been applied.
    """
    columns = ["cleaning_action", "affected_values", "affected_rows", "affected_columns"]
    if not audit_log:
        return pd.DataFrame(columns=columns)

    raw = pd.DataFrame(
        [{"cleaning_action": e.cleaning_action, "row_index": e.row_index, "column_name": e.column_name} for e in audit_log]
    )
    grouped = raw.groupby("cleaning_action").agg(
        affected_values=("cleaning_action", "count"),
        affected_rows=("row_index", "nunique"),
        affected_columns=("column_name", "nunique"),
    ).reset_index()
    return grouped.sort_values("affected_values", ascending=False).reset_index(drop=True)


def unresolved_issues_summary(issues: list[Issue]) -> pd.DataFrame:
    """Group issues that are not safe to auto-fix -- i.e. require human judgment.

    These are not "errors" the system failed to fix; they're findings the
    system deliberately declined to act on without more context (per the
    project's "never assume, always flag" principle).

    Args:
        issues: Issues to summarize (typically the cleaned dataset's
            remaining issues, or the original dataset's issues if no
            cleaning has been applied yet).

    Returns:
        A DataFrame with columns ["issue_type", "issue_category", "count",
        "example_action"], sorted by count descending. Empty if every issue
        is safe to auto-fix (or there are no issues at all).
    """
    columns = ["issue_type", "issue_category", "count", "example_action"]
    unresolved = [issue for issue in issues if not issue.safe_to_auto_fix]
    if not unresolved:
        return pd.DataFrame(columns=columns)

    groups: dict[tuple[str, str], dict] = {}
    for issue in unresolved:
        key = (issue.issue_type, issue.issue_category)
        if key not in groups:
            groups[key] = {
                "issue_type": issue.issue_type,
                "issue_category": issue.issue_category,
                "count": 0,
                "example_action": issue.recommended_action,
            }
        groups[key]["count"] += 1

    result = pd.DataFrame(groups.values())
    return result.sort_values("count", ascending=False).reset_index(drop=True)
