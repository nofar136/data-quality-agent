"""Shared data models for issue detection (Phase 3) and later phases.

Kept separate from the detection logic itself (src/issue_detector.py,
src/rule_engine.py) so the data shapes are stable and easy to reuse once
Phase 4 (cleaning) and Phase 5 (SQL storage) start consuming them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    """Deterministic severity levels. See issue_detector.py for the rules
    that assign each issue type a severity -- never randomly generated."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class IssueCategory(str, Enum):
    """Top-level grouping of issues, mirroring the project specification."""

    DATASET = "Dataset"
    MISSING_VALUE = "Missing Value"
    TYPE = "Type"
    TEXT = "Text"
    NUMERIC = "Numeric"
    DATE = "Date"
    IDENTIFIER = "Identifier"


_SEVERITY_ORDER: list[str] = [Severity.LOW.value, Severity.MEDIUM.value, Severity.HIGH.value, Severity.CRITICAL.value]


def max_severity(a: str, b: str) -> str:
    """Return whichever of two severities is more severe."""
    return a if _SEVERITY_ORDER.index(a) >= _SEVERITY_ORDER.index(b) else b


@dataclass
class Issue:
    """A single detected data quality issue.

    ``row_index`` and ``current_value`` are None for dataset- or
    column-level findings (e.g. "this column is entirely empty") where no
    single row/value applies -- "when available" per the spec.
    """

    run_id: str
    dataset_name: str
    issue_category: str
    issue_type: str
    severity: str
    confidence: float
    recommended_action: str
    rule_name: str
    detected_at: str
    issue_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    column_name: Optional[str] = None
    row_index: Optional[int] = None
    current_value: Optional[str] = None
    suggested_value: Optional[str] = None
    safe_to_auto_fix: bool = False


@dataclass
class DetectionSummary:
    """Aggregate counts over a full set of detected issues."""

    total_issues: int
    by_category: dict[str, int]
    by_severity: dict[str, int]
    by_type: dict[str, int]


@dataclass
class IssueDetectionResult:
    """The full result of running the rule engine once over a dataset."""

    run_id: str
    dataset_name: str
    detected_at: str
    issues: list[Issue]
    summary: DetectionSummary


@dataclass
class AuditLogEntry:
    """A single applied cleaning change (Phase 4).

    ``row_index`` and/or ``column_name`` are None for actions that apply to
    a whole column or the whole dataset (e.g. renaming a column, removing an
    entirely empty column) rather than a specific cell.
    """

    run_id: str
    timestamp: str
    dataset_name: str
    cleaning_action: str
    rule_name: str
    reason: str
    user_approved: bool
    confidence: float
    row_index: Optional[int] = None
    column_name: Optional[str] = None
    original_value: Optional[str] = None
    new_value: Optional[str] = None


@dataclass
class CleaningDecisionLogEntry:
    """A human decision made during guided cleaning (Phase 7B).

    Distinct from ``AuditLogEntry``, which only records an actual value
    change. A decision like "Keep as NULL" or "Do not clean" changes nothing
    in the data but is still recorded here, so the system can tell an
    unresolved issue apart from one a human explicitly reviewed and
    accepted -- an accepted NULL must never be reported as "fixed".

    When a decision *also* changes data (e.g. "Replace with Median"), the
    individual cell changes are additionally recorded in the AuditLogEntry
    list, one entry per changed cell; this entry is the single
    higher-level record of the decision that produced them.
    """

    run_id: str
    timestamp: str
    dataset_name: str
    column_name: str
    issue_type: str
    effective_logical_type: str
    inference_confidence: float
    selected_strategy: str
    scope: str  # "all", "selected", or "n/a" for decisions with no row scope
    affected_count: int
    user_approved: bool
    decision_result: str
    custom_value: Optional[str] = None
    reason: Optional[str] = None
