"""Transparent data quality scoring (Phase 5).

The scorer never re-derives issues itself -- it only aggregates the Issue
records Phase 3 already produced, filtered per component by
``src/config.py``'s ``*_ISSUE_TYPES`` sets. This guarantees the score is
always consistent with whatever the "Data Quality Issues" section shows:
the same detection run drives both.

Five components, each 0-100:

- **Completeness** -- share of cells that are missing (real nulls, blank
  strings, whitespace-only strings, or recognized placeholders).
  Denominator: total cells.
- **Uniqueness** -- share of rows involved in an exact duplicate row or a
  duplicate value in a possible-identifier column (counted once per row
  even if both apply, and normal repeated categorical values are never
  counted at all, since they are not flagged as issues in the first place).
  Denominator: total rows.
- **Validity** -- share of non-null cells that are outright invalid (failed
  type conversion, unexpected text in an otherwise-numeric column, an
  unparseable date, or an infinite value). Denominator: total non-null
  cells.
- **Consistency** -- share of non-null cells with a formatting problem
  (whitespace, capitalization, near-duplicate category spelling, mixed
  date/number formats). Denominator: total non-null cells.
- **Structural Quality** -- count of empty rows, empty columns, duplicate/
  inconsistent column names, and columns stored in the wrong type, relative
  to the dataset's total rows + columns ("structural surface area").

Each component reports the exact issue_count, denominator, and penalty
used, so the score can always be explained rather than trusted blindly.
The overall score is a weighted sum using ``COMPONENT_WEIGHTS`` (sums to
100). Nothing here is random -- given the same issues and the same
dataset shape, the score is always identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.config import (
    COMPLETENESS_ISSUE_TYPES,
    COMPONENT_WEIGHTS,
    CONSISTENCY_ISSUE_TYPES,
    STRUCTURAL_ISSUE_TYPES,
    UNIQUENESS_ISSUE_TYPES,
    VALIDITY_ISSUE_TYPES,
)
from src.models import Issue


@dataclass
class ComponentScore:
    """One component's score, plus everything needed to explain it."""

    component_name: str
    score: float
    issue_count: int
    denominator: int
    penalty: float
    weight: float
    weighted_contribution: float
    explanation: str


@dataclass
class QualityScore:
    """The full quality score for one dataset version (original or cleaned)."""

    dataset_version: str
    overall_score: float
    components: list[ComponentScore] = field(default_factory=list)
    calculated_at: str = ""


def _score_from_ratio(issue_count: int, denominator: int) -> tuple[float, float]:
    """Turn an issue count and denominator into a (score, penalty) pair.

    Args:
        issue_count: Number of flagged occurrences.
        denominator: Total opportunities for that kind of issue.

    Returns:
        (score, penalty), both in [0, 100], where score = 100 - penalty
        and penalty is the percentage of the denominator that was flagged
        (capped at 100 so a component can never go negative).
    """
    if denominator <= 0:
        return 100.0, 0.0
    penalty = min(issue_count / denominator * 100.0, 100.0)
    score = max(100.0 - penalty, 0.0)
    return round(score, 2), round(penalty, 2)


def _component(
    name: str, issue_count: int, denominator: int, explanation_template: str
) -> ComponentScore:
    score, penalty = _score_from_ratio(issue_count, denominator)
    weight = COMPONENT_WEIGHTS[name]
    weighted_contribution = round(score * weight / 100.0, 2)
    explanation = explanation_template.format(issue_count=issue_count, denominator=denominator, penalty=penalty)
    return ComponentScore(
        component_name=name, score=score, issue_count=issue_count, denominator=denominator,
        penalty=penalty, weight=weight, weighted_contribution=weighted_contribution, explanation=explanation,
    )


def _count_by_types(issues: list[Issue], issue_types: frozenset[str]) -> int:
    return sum(1 for issue in issues if issue.issue_type in issue_types)


def _score_completeness(df: pd.DataFrame, issues: list[Issue]) -> ComponentScore:
    total_cells = df.shape[0] * df.shape[1]
    issue_count = _count_by_types(issues, COMPLETENESS_ISSUE_TYPES)
    return _component(
        "Completeness", issue_count, total_cells,
        "{issue_count} of {denominator} cells are missing, blank, or a recognized placeholder ({penalty}% penalty).",
    )


def _score_uniqueness(df: pd.DataFrame, issues: list[Issue]) -> ComponentScore:
    total_rows = df.shape[0]
    affected_rows = {
        issue.row_index for issue in issues
        if issue.issue_type in UNIQUENESS_ISSUE_TYPES and issue.row_index is not None
    }
    return _component(
        "Uniqueness", len(affected_rows), total_rows,
        "{issue_count} of {denominator} rows are an exact duplicate or share a duplicate identifier value "
        "({penalty}% penalty). Normal repeated categorical values are never counted here.",
    )


def _score_validity(df: pd.DataFrame, issues: list[Issue]) -> ComponentScore:
    total_non_null_cells = int(df.notna().sum().sum())
    issue_count = _count_by_types(issues, VALIDITY_ISSUE_TYPES)
    return _component(
        "Validity", issue_count, total_non_null_cells,
        "{issue_count} of {denominator} non-missing cells are invalid (failed conversion, unparseable date, "
        "unexpected text, or infinite) ({penalty}% penalty).",
    )


def _score_consistency(df: pd.DataFrame, issues: list[Issue]) -> ComponentScore:
    total_non_null_cells = int(df.notna().sum().sum())
    issue_count = _count_by_types(issues, CONSISTENCY_ISSUE_TYPES)
    return _component(
        "Consistency", issue_count, total_non_null_cells,
        "{issue_count} of {denominator} non-missing cells have a formatting inconsistency (whitespace, "
        "capitalization, near-duplicate category spelling, or a mixed format) ({penalty}% penalty).",
    )


def _score_structural(df: pd.DataFrame, issues: list[Issue]) -> ComponentScore:
    structural_surface_area = df.shape[0] + df.shape[1]
    issue_count = _count_by_types(issues, STRUCTURAL_ISSUE_TYPES)
    return _component(
        "Structural Quality", issue_count, structural_surface_area,
        "{issue_count} structural problems (empty rows/columns, duplicate/inconsistent column names, "
        "columns stored in the wrong type) relative to {denominator} total rows + columns ({penalty}% penalty).",
    )


def calculate_quality_score(df: pd.DataFrame, issues: list[Issue], dataset_version: str) -> QualityScore:
    """Calculate the five component scores and the overall weighted score.

    Args:
        df: The dataset this score describes (original or cleaned).
        issues: Issue records from src.rule_engine.detect_issues run
            against this same ``df``.
        dataset_version: "original" or "cleaned" -- recorded on the result,
            not used in the calculation itself.

    Returns:
        A QualityScore with all five ComponentScores and the overall score.
    """
    components = [
        _score_completeness(df, issues),
        _score_uniqueness(df, issues),
        _score_validity(df, issues),
        _score_consistency(df, issues),
        _score_structural(df, issues),
    ]
    overall_score = round(sum(c.weighted_contribution for c in components), 2)

    return QualityScore(
        dataset_version=dataset_version,
        overall_score=overall_score,
        components=components,
        calculated_at=datetime.now(timezone.utc).isoformat(),
    )
