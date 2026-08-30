"""Human-in-the-loop logical type override (Phase 7A).

The rule engine (src/rule_engine.py) selects issue checks based on a
column's *effective* logical type, which defaults to the automatically
inferred type but can be overridden by the user for the current session.
Overriding never touches the raw data, and never mutates the inferred
``logical_type``/``confidence``/``evidence`` on a ColumnProfile -- it only
sets ``effective_logical_type`` on a new copy, so the original inference is
always still visible for comparison.

The override UI offers a small, coarse set of types (Numeric, Categorical,
Text, Date / Datetime, Boolean, Identifier, Unknown) rather than the full
LogicalType vocabulary (which distinguishes Integer vs Decimal, Email vs URL
vs Phone, etc.) -- fine-grained distinctions the automatic inference already
handles well are not worth asking a user to manually pick between. Each
coarse option maps onto one representative LogicalType member, which is
sufficient because src/rule_engine.py only ever branches on *groups* of
LogicalType (numeric-like, text-like, date-like, ...), never on the exact
member.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone

from src.profiler import ColumnProfile
from src.schema_inference import LogicalType

USE_DETECTED_LABEL = "Use detected type"

# Display label -> representative LogicalType. Order matters: it's the order
# shown in the Streamlit selectbox.
OVERRIDE_TYPE_OPTIONS: dict[str, LogicalType] = {
    "Numeric": LogicalType.DECIMAL,
    "Currency": LogicalType.CURRENCY,
    "Categorical": LogicalType.CATEGORICAL,
    "Text": LogicalType.FREE_TEXT,
    "Date / Datetime": LogicalType.DATE,
    "Boolean": LogicalType.BOOLEAN,
    "Identifier": LogicalType.IDENTIFIER,
    "Email": LogicalType.EMAIL,
    "Phone": LogicalType.PHONE,
    "URL": LogicalType.URL,
    "Leave as Unknown": LogicalType.UNKNOWN,
}

OVERRIDE_LABELS: tuple[str, ...] = (USE_DETECTED_LABEL, *OVERRIDE_TYPE_OPTIONS.keys())


@dataclass
class TypeOverrideRecord:
    """An auditable record of one user decision about a column's type.

    Created whether the user confirmed the detected type or changed it --
    ``user_approved`` is always True (the user explicitly acted), and
    ``original_type == new_type`` when they simply confirmed rather than
    changed it.
    """

    column_name: str
    original_type: str
    original_confidence: float
    new_type: str
    user_approved: bool
    timestamp: str


def is_low_confidence(confidence: float, threshold: float) -> bool:
    """Whether an inference confidence falls below the review-warning threshold."""
    return confidence < threshold


def make_override_record(profile: ColumnProfile, selected_label: str) -> TypeOverrideRecord:
    """Build the audit record for a user's type decision on one column.

    Args:
        profile: The column's profile (its inferred type/confidence are the "original").
        selected_label: What the user picked in the selectbox -- either
            USE_DETECTED_LABEL (a confirmation, not a change) or one of
            OVERRIDE_TYPE_OPTIONS's keys.

    Returns:
        A TypeOverrideRecord describing this decision.
    """
    new_type = profile.logical_type if selected_label == USE_DETECTED_LABEL else OVERRIDE_TYPE_OPTIONS[selected_label].value
    return TypeOverrideRecord(
        column_name=profile.original_name,
        original_type=profile.logical_type,
        original_confidence=profile.confidence,
        new_type=new_type,
        user_approved=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def apply_type_overrides(profiles: list[ColumnProfile], overrides: dict[str, str]) -> list[ColumnProfile]:
    """Return new ColumnProfiles with effective_logical_type set from any overrides.

    Never mutates the input profiles or touches raw data -- this only
    changes which issue checks src/rule_engine.py runs for the current
    session.

    Args:
        profiles: Profiles from src.profiler.profile_dataframe (unmodified).
        overrides: {column_name: selected_label}, where selected_label is
            USE_DETECTED_LABEL or a key of OVERRIDE_TYPE_OPTIONS. Columns
            with no entry (or USE_DETECTED_LABEL) are left at their inferred type.

    Returns:
        A new list of ColumnProfile, same order, with effective_logical_type,
        type_overridden, and user_selected_type updated per the overrides.
    """
    result: list[ColumnProfile] = []
    for profile in profiles:
        label = overrides.get(profile.original_name, USE_DETECTED_LABEL)
        if label == USE_DETECTED_LABEL or label not in OVERRIDE_TYPE_OPTIONS:
            result.append(replace(profile, effective_logical_type=profile.logical_type, type_overridden=False, user_selected_type=None))
        else:
            effective_type = OVERRIDE_TYPE_OPTIONS[label].value
            result.append(replace(profile, effective_logical_type=effective_type, type_overridden=True, user_selected_type=label))
    return result
