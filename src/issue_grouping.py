"""Groups detected issues into guided-review units (Phase 7B).

The Data Cleaning page never lists individual issues one row at a time --
it groups them by (column, issue_type) so a user reviews "24 missing values
in Salary" as one card, not 24 separate rows. Only issue types that the
Cleaning Strategy Engine (src/cleaning_strategies.py) actually has
strategies for are surfaced as guided-review groups; everything else stays
visible only on the Data Quality Issues page.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.models import Issue
from src.profiler import ColumnProfile

# Issue types the guided cleaning workflow offers a strategy for. Structural/
# text-hygiene issues (whitespace, placeholders, duplicate rows, ...) are
# already handled by the existing automatic safe fixes (Phase 4/6) and are
# intentionally not repeated here.
GUIDED_REVIEW_ISSUE_TYPES: frozenset[str] = frozenset(
    {"missing_null", "possible_outlier", "negative_value", "similar_category_values", "inconsistent_capitalization"}
)

_SEVERITY_RANK: dict[str, int] = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}


@dataclass
class IssueGroup:
    """One (column, issue_type) cluster of issues to review together."""

    column_name: str
    issue_type: str
    issue_category: str
    severity: str
    effective_logical_type: str
    inference_confidence: float
    recommended_action: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def affected_count(self) -> int:
        return len(self.issues)

    @property
    def row_indices(self) -> list[int]:
        return sorted({i.row_index for i in self.issues if i.row_index is not None})


def build_issue_groups(issues: list[Issue], profiles_by_name: dict[str, ColumnProfile]) -> list[IssueGroup]:
    """Group issues by (column, issue_type), keeping only guided-review-eligible types.

    Args:
        issues: Issues detected on the current working copy (see
            src.rule_engine.detect_issues).
        profiles_by_name: {original_name: ColumnProfile} for the same
            DataFrame the issues were detected on, used to attach each
            group's effective logical type and confidence.

    Returns:
        One IssueGroup per (column, issue_type) with at least one
        guided-review-eligible issue, sorted by severity (most severe
        first) and then by affected count (largest first).
    """
    buckets: dict[tuple[str, str], list[Issue]] = {}
    for issue in issues:
        if issue.issue_type not in GUIDED_REVIEW_ISSUE_TYPES or not issue.column_name:
            continue
        key = (issue.column_name, issue.issue_type)
        buckets.setdefault(key, []).append(issue)

    groups: list[IssueGroup] = []
    for (column_name, issue_type), group_issues in buckets.items():
        profile = profiles_by_name.get(column_name)
        groups.append(
            IssueGroup(
                column_name=column_name,
                issue_type=issue_type,
                issue_category=group_issues[0].issue_category,
                severity=group_issues[0].severity,
                effective_logical_type=profile.effective_logical_type if profile else "Unknown",
                inference_confidence=profile.confidence if profile else 0.0,
                recommended_action=group_issues[0].recommended_action,
                issues=group_issues,
            )
        )

    groups.sort(key=lambda g: (-_SEVERITY_RANK.get(g.severity, 0), -g.affected_count))
    return groups
