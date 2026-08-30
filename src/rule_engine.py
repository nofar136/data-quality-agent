"""Generic rule engine (Phase 3).

Selects which issue-detection checks apply to each column based purely on
its inferred logical type (from src/schema_inference.py via src/profiler.py)
-- never on column name or position. Dataset-level checks always run.

This is the single entry point issue detection should be called through;
src/issue_detector.py implements each individual check.
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from src.issue_detector import (
    detect_dataset_level_issues,
    detect_date_issues,
    detect_identifier_issues,
    detect_missing_value_issues,
    detect_numeric_issues,
    detect_text_issues,
    detect_type_issues,
)
from src.models import DetectionSummary, Issue, IssueDetectionResult
from src.profiler import ColumnProfile
from src.schema_inference import LogicalType

_TEXT_LIKE_TYPES = {
    LogicalType.CATEGORICAL.value,
    LogicalType.FREE_TEXT.value,
    LogicalType.EMAIL.value,
    LogicalType.URL.value,
    LogicalType.PHONE.value,
    LogicalType.IDENTIFIER.value,
    LogicalType.MIXED.value,
    # UNKNOWN only arises from the text/object branch of type inference (numeric,
    # date, boolean, and empty columns all resolve via their own dtype fast paths
    # first) -- so it is still text data that can have whitespace/formatting
    # problems, even though its semantic type couldn't be confidently determined.
    LogicalType.UNKNOWN.value,
}
_NUMERIC_LIKE_TYPES = {
    LogicalType.INTEGER.value,
    LogicalType.DECIMAL.value,
    LogicalType.NUMERIC_TEXT.value,
    LogicalType.CURRENCY.value,
}
_DATE_LIKE_TYPES = {LogicalType.DATE.value, LogicalType.DATETIME.value, LogicalType.DATE_TEXT.value}
_TYPE_ISSUE_TYPES = {LogicalType.NUMERIC_TEXT.value, LogicalType.DATE_TEXT.value, LogicalType.MIXED.value}


def _build_summary(issues: list[Issue]) -> DetectionSummary:
    return DetectionSummary(
        total_issues=len(issues),
        by_category=dict(Counter(issue.issue_category for issue in issues)),
        by_severity=dict(Counter(issue.severity for issue in issues)),
        by_type=dict(Counter(issue.issue_type for issue in issues)),
    )


def detect_issues(df: pd.DataFrame, profiles: list[ColumnProfile], dataset_name: str) -> IssueDetectionResult:
    """Run every applicable data quality check over a dataset.

    Args:
        df: The dataset to check.
        profiles: Column profiles from src.profiler.profile_dataframe(df) --
            used to decide which column-level checks apply and to reuse
            already-computed type inference.
        dataset_name: Name to record on every issue (e.g. the uploaded file name).

    Returns:
        An IssueDetectionResult with the full, uncapped list of issues and
        an accurate summary (total / by category / by severity / by type).
    """
    run_id = str(uuid.uuid4())
    detected_at = datetime.now(timezone.utc).isoformat()

    issues: list[Issue] = list(detect_dataset_level_issues(df, dataset_name, run_id, detected_at))

    profiles_by_name = {profile.original_name: profile for profile in profiles}

    for column in df.columns:
        profile = profiles_by_name[str(column)]
        series = df[column]
        # Uses the *effective* type (inferred, unless the user overrode it
        # for this session via src/type_override.py) -- never the raw
        # dtype, and never forced to match a user override that wasn't made.
        logical_type = profile.effective_logical_type

        issues.extend(detect_missing_value_issues(series, profile, dataset_name, run_id, detected_at))

        if logical_type in _TYPE_ISSUE_TYPES:
            issues.extend(detect_type_issues(series, profile, dataset_name, run_id, detected_at))

        if logical_type in _TEXT_LIKE_TYPES:
            issues.extend(detect_text_issues(series, profile, dataset_name, run_id, detected_at))

        if logical_type in _NUMERIC_LIKE_TYPES:
            issues.extend(detect_numeric_issues(series, profile, dataset_name, run_id, detected_at))

        if logical_type in _DATE_LIKE_TYPES:
            issues.extend(detect_date_issues(series, profile, dataset_name, run_id, detected_at))

        if logical_type == LogicalType.IDENTIFIER.value:
            issues.extend(detect_identifier_issues(series, profile, dataset_name, run_id, detected_at))

    return IssueDetectionResult(
        run_id=run_id,
        dataset_name=dataset_name,
        detected_at=detected_at,
        issues=issues,
        summary=_build_summary(issues),
    )
