"""Tests for src.database.

All tests use a temporary SQLite file (pytest's tmp_path fixture) --
never the real database/data_quality.db.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.database import (
    AppliedFixSummary,
    DatabaseError,
    RunBundle,
    get_connection,
    get_run_detail,
    get_run_history,
    init_db,
    run_exists,
    save_run,
)
from src.profiler import profile_dataframe
from src.rule_engine import detect_issues
from src.scoring import calculate_quality_score


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_data_quality.db"


def _make_bundle(run_id: str = "run-1", dataset_name: str = "sample.csv", with_cleaning: bool = False) -> RunBundle:
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["  Alice", "Bob", "Carol"]})
    profiles = profile_dataframe(df)
    issues = detect_issues(df, profiles, dataset_name).issues
    score = calculate_quality_score(df, issues, "original")

    kwargs = dict(
        run_id=run_id, dataset_name=dataset_name, file_type="csv", selected_sheet=None,
        run_timestamp="2024-01-01T00:00:00+00:00", original_shape=df.shape,
        original_profiles=profiles, original_issues=issues, original_score=score,
    )

    if with_cleaning:
        cleaned_df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})
        cleaned_profiles = profile_dataframe(cleaned_df)
        cleaned_issues = detect_issues(cleaned_df, cleaned_profiles, dataset_name).issues
        cleaned_score = calculate_quality_score(cleaned_df, cleaned_issues, "cleaned")
        kwargs.update(
            cleaned_shape=cleaned_df.shape, cleaned_profiles=cleaned_profiles,
            cleaned_issues=cleaned_issues, cleaned_score=cleaned_score,
            audit_log=[], applied_fixes=[AppliedFixSummary("trim_whitespace", "Trim whitespace", 2)],
        )

    return RunBundle(**kwargs)


# --- Database initialization ---------------------------------------------------------


def test_init_db_creates_all_expected_tables(db_path: Path) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    expected = {
        "schema_info", "profiling_runs", "dataset_profiles", "column_profiles",
        "data_quality_issues", "cleaning_audit_log", "applied_fixes", "quality_scores",
    }
    assert expected <= tables


def test_init_db_is_safe_to_run_repeatedly(db_path: Path) -> None:
    init_db(db_path)
    init_db(db_path)  # must not raise
    with get_connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM schema_info").fetchone()[0]
    assert count == 1  # INSERT OR IGNORE keeps this at one row per schema_version


# --- Foreign-key enforcement -----------------------------------------------------------


def test_foreign_keys_are_enforced(db_path: Path) -> None:
    init_db(db_path)
    with pytest.raises(sqlite3.IntegrityError):
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO column_profiles (run_id, dataset_version, column_name, normalized_name, "
                "pandas_dtype, logical_type, confidence, non_null_count, missing_count, missing_pct, "
                "unique_count, unique_ratio) VALUES ('nonexistent-run', 'original', 'x', 'x', 'object', "
                "'Unknown', 1.0, 1, 0, 0.0, 1, 1.0)"
            )


# --- Saving a complete run ------------------------------------------------------------------


def test_save_run_persists_all_expected_data(db_path: Path) -> None:
    bundle = _make_bundle(with_cleaning=True)
    save_run(bundle, db_path)

    detail = get_run_detail(db_path, bundle.run_id)
    assert len(detail["run"]) == 1
    assert len(detail["column_profiles"]) == 4  # 2 columns x (original + cleaned)
    assert set(detail["issues"]["dataset_version"]) <= {"original", "cleaned"}
    assert len(detail["scores"]) == 10  # 5 components x (original + cleaned)
    assert len(detail["applied_fixes"]) == 1


def test_save_run_without_cleaning_only_stores_original(db_path: Path) -> None:
    bundle = _make_bundle(with_cleaning=False)
    save_run(bundle, db_path)

    detail = get_run_detail(db_path, bundle.run_id)
    assert set(detail["column_profiles"]["dataset_version"]) == {"original"}
    assert set(detail["scores"]["dataset_version"]) == {"original"}
    assert detail["applied_fixes"].empty


# --- Transaction rollback on failure ---------------------------------------------------------


def test_transaction_rolls_back_on_failure(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _make_bundle()

    import src.database as database_module

    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError("simulated failure")

    monkeypatch.setattr(database_module, "_insert_scores", _boom)

    with pytest.raises(DatabaseError):
        save_run(bundle, db_path)

    # Nothing from this run should have been committed.
    assert run_exists(db_path, bundle.run_id) is False
    detail = get_run_detail(db_path, bundle.run_id)
    assert detail["column_profiles"].empty


# --- Duplicate run prevention -----------------------------------------------------------------


def test_saving_the_same_run_twice_raises_and_keeps_original(db_path: Path) -> None:
    bundle = _make_bundle()
    save_run(bundle, db_path)

    with pytest.raises(DatabaseError, match="already exists"):
        save_run(bundle, db_path)

    history = get_run_history(db_path)
    assert len(history) == 1


# --- Run history / comparison ------------------------------------------------------------------


def test_run_history_is_empty_when_nothing_saved(db_path: Path) -> None:
    init_db(db_path)
    history = get_run_history(db_path)
    assert history.empty


def test_loading_run_history_after_saving(db_path: Path) -> None:
    save_run(_make_bundle(run_id="run-a", dataset_name="a.csv"), db_path)
    save_run(_make_bundle(run_id="run-b", dataset_name="b.csv", with_cleaning=True), db_path)

    history = get_run_history(db_path)
    assert len(history) == 2
    assert set(history["run_id"]) == {"run-a", "run-b"}


def test_comparing_original_and_cleaned_scores_in_history(db_path: Path) -> None:
    save_run(_make_bundle(run_id="run-cleaned", with_cleaning=True), db_path)
    history = get_run_history(db_path)
    row = history.loc[history["run_id"] == "run-cleaned"].iloc[0]

    assert row["cleaning_applied"] == True  # noqa: E712
    assert pd.notna(row["original_score"])
    assert pd.notna(row["cleaned_score"])
    assert row["score_improvement"] == pytest.approx(row["cleaned_score"] - row["original_score"], abs=0.01)


def test_history_works_when_no_cleaned_dataset_exists(db_path: Path) -> None:
    save_run(_make_bundle(run_id="run-no-clean", with_cleaning=False), db_path)
    history = get_run_history(db_path)
    row = history.loc[history["run_id"] == "run-no-clean"].iloc[0]

    assert row["cleaning_applied"] == False  # noqa: E712
    assert pd.isna(row["cleaned_score"])


# --- Original uploaded dataset remains unchanged -------------------------------------------------


def test_save_run_never_stores_raw_dataset_rows(db_path: Path) -> None:
    bundle = _make_bundle(with_cleaning=True)
    save_run(bundle, db_path)

    with get_connection(db_path) as conn:
        table_names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    # No table should contain a full copy of the dataset's actual row cells
    # (only issue/audit records that reference specific cells, not a bulk dump).
    forbidden_table_names = {"dataset_rows", "raw_data", "uploaded_data"}
    assert not (forbidden_table_names & table_names)
