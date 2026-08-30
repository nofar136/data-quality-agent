"""Tests for src.scoring."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import (
    COMPLETENESS_ISSUE_TYPES,
    COMPONENT_WEIGHTS,
    CONSISTENCY_ISSUE_TYPES,
    SCORING_EXCLUDED_ISSUE_TYPES,
    STRUCTURAL_ISSUE_TYPES,
    UNIQUENESS_ISSUE_TYPES,
    VALIDITY_ISSUE_TYPES,
)
from src.issue_detector import ISSUE_TYPE_EXPLANATIONS
from src.profiler import profile_dataframe
from src.rule_engine import detect_issues
from src.scoring import calculate_quality_score


def _score(df: pd.DataFrame, dataset_name: str = "ds"):
    issues = detect_issues(df, profile_dataframe(df), dataset_name).issues
    return calculate_quality_score(df, issues, "original")


# --- Component set hygiene ----------------------------------------------------------


def test_component_issue_type_sets_are_pairwise_disjoint() -> None:
    sets = [COMPLETENESS_ISSUE_TYPES, UNIQUENESS_ISSUE_TYPES, VALIDITY_ISSUE_TYPES, CONSISTENCY_ISSUE_TYPES, STRUCTURAL_ISSUE_TYPES]
    for i, set_a in enumerate(sets):
        for set_b in sets[i + 1 :]:
            assert not (set_a & set_b), f"overlap found: {set_a & set_b}"


def test_every_known_issue_type_is_classified() -> None:
    all_known_types = set(ISSUE_TYPE_EXPLANATIONS.keys())
    classified = COMPLETENESS_ISSUE_TYPES | UNIQUENESS_ISSUE_TYPES | VALIDITY_ISSUE_TYPES | CONSISTENCY_ISSUE_TYPES | STRUCTURAL_ISSUE_TYPES | SCORING_EXCLUDED_ISSUE_TYPES
    assert all_known_types <= classified, f"unclassified: {all_known_types - classified}"


def test_weights_sum_to_100() -> None:
    assert sum(COMPONENT_WEIGHTS.values()) == pytest.approx(100.0)


# --- Individual component scoring -----------------------------------------------------


def test_completeness_scoring_penalizes_missing_values() -> None:
    df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
    score = _score(df)
    completeness = next(c for c in score.components if c.component_name == "Completeness")

    assert completeness.issue_count == 2
    assert completeness.denominator == 8  # 4 rows * 2 cols
    assert completeness.score == pytest.approx(75.0)


def test_completeness_perfect_when_no_missing_values() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    score = _score(df)
    completeness = next(c for c in score.components if c.component_name == "Completeness")
    assert completeness.score == 100.0
    assert completeness.issue_count == 0


def test_uniqueness_scoring_counts_duplicate_rows() -> None:
    df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    score = _score(df)
    uniqueness = next(c for c in score.components if c.component_name == "Uniqueness")

    assert uniqueness.issue_count == 1
    assert uniqueness.denominator == 3
    assert uniqueness.score == pytest.approx((1 - 1 / 3) * 100, abs=0.01)


def test_uniqueness_does_not_penalize_repeated_categorical_values() -> None:
    df = pd.DataFrame(
        {
            "id": [f"ID-{i:03d}" for i in range(20)],
            "category": (["Electronics", "Groceries", "Books", "Clothing"] * 5),
        }
    )
    score = _score(df)
    uniqueness = next(c for c in score.components if c.component_name == "Uniqueness")
    assert uniqueness.score == 100.0
    assert uniqueness.issue_count == 0


def test_validity_scoring_counts_invalid_values() -> None:
    df = pd.DataFrame({"amount": [str(i) for i in range(9)] + ["abc"]})
    score = _score(df)
    validity = next(c for c in score.components if c.component_name == "Validity")
    assert validity.issue_count == 1
    assert validity.denominator == 10


def test_consistency_scoring_counts_formatting_issues() -> None:
    df = pd.DataFrame({"name": ["  Alice", "Bob", "Carol"]})
    score = _score(df)
    consistency = next(c for c in score.components if c.component_name == "Consistency")
    assert consistency.issue_count == 1
    assert consistency.denominator == 3


def test_structural_scoring_counts_empty_columns_and_rows() -> None:
    df = pd.DataFrame({"a": [1, None], "b": [None, None]})
    score = _score(df)
    structural = next(c for c in score.components if c.component_name == "Structural Quality")
    assert structural.issue_count >= 2  # empty_row (row 1) + empty_column (b)
    assert structural.denominator == df.shape[0] + df.shape[1]


# --- Overall score ------------------------------------------------------------------------


def test_overall_score_is_weighted_sum_of_components() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    score = _score(df)
    expected = round(sum(c.weighted_contribution for c in score.components), 2)
    assert score.overall_score == expected


def test_all_scores_stay_within_0_and_100() -> None:
    df = pd.DataFrame(
        {
            "a": [None, None, None, "abc", "1"],
            "b": ["", "", "N/A", "x", "x"],
        }
    )
    score = _score(df)
    assert 0.0 <= score.overall_score <= 100.0
    for component in score.components:
        assert 0.0 <= component.score <= 100.0


# --- Clean vs dirty datasets ----------------------------------------------------------------


def test_clean_dataset_scores_high() -> None:
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:04d}" for i in range(50)],
            "email": [f"user{i}@example.com" for i in range(50)],
            "amount": [round(9.99 + i, 2) for i in range(50)],
            "category": (["Electronics", "Groceries", "Clothing", "Books", "Toys"] * 10),
            "signup_date": pd.date_range("2023-01-01", periods=50, freq="D"),
        }
    )
    score = _score(df)
    assert score.overall_score >= 95.0


def test_dirty_dataset_scores_lower_than_clean_dataset() -> None:
    clean_df = pd.DataFrame(
        {
            "id": [f"ID-{i:04d}" for i in range(30)],
            "amount": [round(9.99 + i, 2) for i in range(30)],
        }
    )
    dirty_df = pd.DataFrame(
        {
            "id": [f"ID-{i:04d}" for i in range(29)] + ["ID-0000"],
            "amount": [str(i) for i in range(28)] + ["abc", ""],
        }
    )
    assert _score(dirty_df).overall_score < _score(clean_df).overall_score


def test_outliers_do_not_reduce_the_score() -> None:
    # A unique id column ensures no row is an exact duplicate of another, so
    # the only difference between the two datasets is the outlier itself.
    df_without_outlier = pd.DataFrame({"id": range(8), "value": [10, 11, 9, 10, 12, 11, 9, 10]})
    df_with_outlier = pd.DataFrame({"id": range(8), "value": [10, 11, 9, 10, 12, 11, 9, 1000]})

    assert _score(df_without_outlier).overall_score == _score(df_with_outlier).overall_score


def test_negative_values_do_not_reduce_the_score() -> None:
    # Phase 7B: negative values are surfaced for review in the Guided Issue
    # Review workflow, but never penalized by default -- before any user
    # decision, the score must be identical with or without negatives.
    df_without_negatives = pd.DataFrame({"id": range(8), "delta": [10, 11, 9, 10, 12, 11, 9, 10]})
    df_with_negatives = pd.DataFrame({"id": range(8), "delta": [10, 11, 9, 10, 12, 11, 9, -10]})

    assert _score(df_without_negatives).overall_score == _score(df_with_negatives).overall_score


# --- Cleaning improves the score ------------------------------------------------------------


def test_score_improves_after_fixing_a_real_issue() -> None:
    dirty_df = pd.DataFrame({"name": ["  Alice", "Bob  ", "Carol"]})
    cleaned_df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"]})

    assert _score(cleaned_df).overall_score > _score(dirty_df).overall_score
