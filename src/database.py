"""SQLite persistence for profiling runs (Phase 5).

Only profiles, detected issues, cleaning audit entries, applied-fix
summaries, and quality scores are stored -- never the uploaded or cleaned
dataset itself (see sql/create_tables.sql for the schema and rationale).

Saving is always explicit: nothing in this module is called automatically
on every Streamlit rerun. The caller (app.py) decides when a run is
actually persisted, in response to a user clicking "Save Run to Database".

Every write happens inside a single transaction per run (via
``connection_scope``, which wraps sqlite3's idiomatic auto-commit/
auto-rollback ``with conn:`` block and also guarantees the connection is
closed), so a failure partway through never leaves a partially saved run
behind.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from src.config import CREATE_TABLES_SQL_PATH, DATABASE_PATH, SCHEMA_VERSION
from src.models import AuditLogEntry, Issue
from src.profiler import ColumnProfile
from src.scoring import QualityScore


class DatabaseError(Exception):
    """Raised for any database failure, with a message safe to show a user."""


@dataclass
class AppliedFixSummary:
    """One rollup row: how many changes one fix type made in a run."""

    fix_id: str
    fix_title: str
    affected_count: int


@dataclass
class RunBundle:
    """Everything needed to persist one profiling/cleaning run.

    The ``cleaned_*`` fields are all optional together: if cleaning was not
    performed, pass None for all of them and ``cleaning_applied`` stays False.
    """

    run_id: str
    dataset_name: str
    file_type: str
    selected_sheet: Optional[str]
    run_timestamp: str
    original_shape: tuple[int, int]
    original_profiles: list[ColumnProfile]
    original_issues: list[Issue]
    original_score: QualityScore
    cleaned_shape: Optional[tuple[int, int]] = None
    cleaned_profiles: Optional[list[ColumnProfile]] = None
    cleaned_issues: Optional[list[Issue]] = None
    cleaned_score: Optional[QualityScore] = None
    audit_log: Optional[list[AuditLogEntry]] = None
    applied_fixes: Optional[list[AppliedFixSummary]] = None

    @property
    def cleaning_applied(self) -> bool:
        return self.cleaned_shape is not None


def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign-key enforcement enabled.

    Args:
        db_path: Path to the .db file. Parent directory is created if missing.

    Returns:
        An open connection. Caller is responsible for closing it.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connection_scope(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback as a transaction, and always close it.

    sqlite3.Connection's own context manager protocol only commits or rolls
    back -- it does NOT close the connection. Using ``with get_connection():``
    directly leaks a connection (and, on Windows, a file lock) every time.
    This wraps that behavior so every caller gets both.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply additive, idempotent schema migrations for existing databases.

    CREATE TABLE IF NOT EXISTS never alters a table that already exists, so
    a column added after a database was first created (e.g. Phase 6's
    ``outlier_count``) needs an explicit, safe-to-repeat ALTER TABLE here.
    """
    if not _column_exists(conn, "column_profiles", "outlier_count"):
        conn.execute("ALTER TABLE column_profiles ADD COLUMN outlier_count INTEGER")


def init_db(db_path: Path = DATABASE_PATH) -> None:
    """Create all tables/indexes if missing, and apply any pending migrations.

    Safe to call every time the app starts -- every statement in
    create_tables.sql is idempotent (CREATE TABLE/INDEX IF NOT EXISTS), and
    migrations check for their target column before altering anything.

    Args:
        db_path: Path to the .db file.

    Raises:
        DatabaseError: If the schema script cannot be read or executed.
    """
    try:
        sql_script = CREATE_TABLES_SQL_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatabaseError(f"Could not read schema file at {CREATE_TABLES_SQL_PATH}: {exc}") from exc

    try:
        with connection_scope(db_path) as conn:
            conn.executescript(sql_script)
            _apply_migrations(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_info (schema_version, applied_at) VALUES (?, datetime('now'))",
                (SCHEMA_VERSION,),
            )
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to initialize the database: {exc}") from exc


def run_exists(db_path: Path, run_id: str) -> bool:
    """Check whether a run has already been saved.

    Args:
        db_path: Path to the .db file.
        run_id: The run to check.

    Returns:
        True if a profiling_runs row with this run_id already exists.
    """
    with connection_scope(db_path) as conn:
        row = conn.execute("SELECT 1 FROM profiling_runs WHERE run_id = ?", (run_id,)).fetchone()
    return row is not None


def _insert_dataset_profile(conn: sqlite3.Connection, run_id: str, version: str, df_shape: tuple[int, int], issues: list[Issue], score: QualityScore, duplicate_row_count: int, empty_row_count: int, empty_column_count: int, total_missing_count: int) -> None:
    row_count, column_count = df_shape
    total_cells = row_count * column_count
    missing_pct = round((total_missing_count / total_cells * 100), 2) if total_cells else 0.0
    conn.execute(
        """
        INSERT INTO dataset_profiles (
            run_id, dataset_version, row_count, column_count, total_cells,
            duplicate_row_count, empty_row_count, empty_column_count,
            total_missing_count, missing_pct, overall_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, version, row_count, column_count, total_cells, duplicate_row_count, empty_row_count, empty_column_count, total_missing_count, missing_pct, score.overall_score),
    )


def _insert_column_profiles(conn: sqlite3.Connection, run_id: str, version: str, profiles: list[ColumnProfile]) -> None:
    conn.executemany(
        """
        INSERT INTO column_profiles (
            run_id, dataset_version, column_name, normalized_name, pandas_dtype,
            logical_type, confidence, non_null_count, missing_count, missing_pct,
            unique_count, unique_ratio, outlier_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (run_id, version, p.original_name, p.normalized_name, p.pandas_dtype, p.logical_type, p.confidence, p.non_null_count, p.missing_count, p.missing_pct, p.unique_count, p.unique_ratio, p.outlier_count)
            for p in profiles
        ],
    )


def _insert_issues(conn: sqlite3.Connection, run_id: str, version: str, issues: list[Issue]) -> None:
    conn.executemany(
        """
        INSERT INTO data_quality_issues (
            run_id, dataset_version, issue_id, column_name, row_index, issue_category,
            issue_type, severity, confidence, current_value, suggested_value,
            recommended_action, safe_to_auto_fix, rule_name, detected_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (run_id, version, i.issue_id, i.column_name, i.row_index, i.issue_category, i.issue_type, i.severity, i.confidence, i.current_value, i.suggested_value, i.recommended_action, int(i.safe_to_auto_fix), i.rule_name, i.detected_at)
            for i in issues
        ],
    )


def _insert_scores(conn: sqlite3.Connection, run_id: str, version: str, score: QualityScore) -> None:
    conn.executemany(
        """
        INSERT INTO quality_scores (
            run_id, dataset_version, component_name, component_score, issue_count,
            denominator, penalty, component_weight, weighted_contribution,
            calculation_explanation, calculated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (run_id, version, c.component_name, c.score, c.issue_count, c.denominator, c.penalty, c.weight, c.weighted_contribution, c.explanation, score.calculated_at)
            for c in score.components
        ],
    )


def _insert_audit_log(conn: sqlite3.Connection, run_id: str, entries: list[AuditLogEntry]) -> None:
    conn.executemany(
        """
        INSERT INTO cleaning_audit_log (
            run_id, timestamp, dataset_name, row_index, column_name, original_value,
            new_value, cleaning_action, rule_name, reason, user_approved, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (run_id, e.timestamp, e.dataset_name, e.row_index, e.column_name, e.original_value, e.new_value, e.cleaning_action, e.rule_name, e.reason, int(e.user_approved), e.confidence)
            for e in entries
        ],
    )


def _insert_applied_fixes(conn: sqlite3.Connection, run_id: str, fixes: list[AppliedFixSummary]) -> None:
    conn.executemany(
        "INSERT INTO applied_fixes (run_id, fix_id, fix_title, affected_count) VALUES (?, ?, ?, ?)",
        [(run_id, f.fix_id, f.fix_title, f.affected_count) for f in fixes],
    )


def save_run(bundle: RunBundle, db_path: Path = DATABASE_PATH) -> None:
    """Persist a full profiling/cleaning run in a single transaction.

    Args:
        bundle: Everything to save (see RunBundle).
        db_path: Path to the .db file.

    Raises:
        DatabaseError: If the run_id already exists, or any other database
            error occurs. Nothing is left partially saved either way.
    """
    init_db(db_path)

    original_row_count, original_column_count = bundle.original_shape
    cleaned_row_count, cleaned_column_count = bundle.cleaned_shape if bundle.cleaned_shape else (None, None)

    try:
        with connection_scope(db_path) as conn:
            conn.execute(
                """
                INSERT INTO profiling_runs (
                    run_id, dataset_name, file_type, selected_sheet, run_timestamp,
                    original_row_count, original_column_count, cleaned_row_count,
                    cleaned_column_count, cleaning_applied, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bundle.run_id, bundle.dataset_name, bundle.file_type, bundle.selected_sheet, bundle.run_timestamp,
                    original_row_count, original_column_count, cleaned_row_count, cleaned_column_count,
                    int(bundle.cleaning_applied), SCHEMA_VERSION,
                ),
            )

            original_duplicate_rows = sum(1 for i in bundle.original_issues if i.issue_type == "exact_duplicate_row")
            original_empty_rows = sum(1 for i in bundle.original_issues if i.issue_type == "empty_row")
            original_empty_columns = sum(1 for i in bundle.original_issues if i.issue_type == "empty_column")
            original_missing = sum(
                1 for i in bundle.original_issues
                if i.issue_type in ("missing_null", "blank_string", "whitespace_only_string", "missing_placeholder")
            )
            _insert_dataset_profile(
                conn, bundle.run_id, "original", bundle.original_shape, bundle.original_issues, bundle.original_score,
                original_duplicate_rows, original_empty_rows, original_empty_columns, original_missing,
            )
            _insert_column_profiles(conn, bundle.run_id, "original", bundle.original_profiles)
            _insert_issues(conn, bundle.run_id, "original", bundle.original_issues)
            _insert_scores(conn, bundle.run_id, "original", bundle.original_score)

            if bundle.cleaning_applied:
                cleaned_duplicate_rows = sum(1 for i in bundle.cleaned_issues if i.issue_type == "exact_duplicate_row")
                cleaned_empty_rows = sum(1 for i in bundle.cleaned_issues if i.issue_type == "empty_row")
                cleaned_empty_columns = sum(1 for i in bundle.cleaned_issues if i.issue_type == "empty_column")
                cleaned_missing = sum(
                    1 for i in bundle.cleaned_issues
                    if i.issue_type in ("missing_null", "blank_string", "whitespace_only_string", "missing_placeholder")
                )
                _insert_dataset_profile(
                    conn, bundle.run_id, "cleaned", bundle.cleaned_shape, bundle.cleaned_issues, bundle.cleaned_score,
                    cleaned_duplicate_rows, cleaned_empty_rows, cleaned_empty_columns, cleaned_missing,
                )
                _insert_column_profiles(conn, bundle.run_id, "cleaned", bundle.cleaned_profiles)
                _insert_issues(conn, bundle.run_id, "cleaned", bundle.cleaned_issues)
                _insert_scores(conn, bundle.run_id, "cleaned", bundle.cleaned_score)

                if bundle.audit_log:
                    _insert_audit_log(conn, bundle.run_id, bundle.audit_log)
                if bundle.applied_fixes:
                    _insert_applied_fixes(conn, bundle.run_id, bundle.applied_fixes)

    except sqlite3.IntegrityError as exc:
        raise DatabaseError(f"Run '{bundle.run_id}' already exists in the database -- it was not saved again.") from exc
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to save run '{bundle.run_id}': {exc}") from exc


def get_run_history(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Load a summary row per saved run, newest first.

    Args:
        db_path: Path to the .db file.

    Returns:
        A DataFrame with one row per run (empty if none saved yet). Columns:
        run_id, dataset_name, file_type, run_timestamp, cleaning_applied,
        row/column counts, original_score, cleaned_score, score_improvement,
        issue counts before/after, fixes_applied_count.
    """
    query = """
        SELECT
            pr.run_id, pr.dataset_name, pr.file_type, pr.run_timestamp, pr.cleaning_applied,
            pr.original_row_count, pr.original_column_count,
            pr.cleaned_row_count, pr.cleaned_column_count,
            orig.overall_score AS original_score,
            clean.overall_score AS cleaned_score,
            (SELECT COUNT(*) FROM data_quality_issues WHERE run_id = pr.run_id AND dataset_version = 'original') AS original_issue_count,
            (SELECT COUNT(*) FROM data_quality_issues WHERE run_id = pr.run_id AND dataset_version = 'cleaned') AS cleaned_issue_count,
            (SELECT COUNT(*) FROM applied_fixes WHERE run_id = pr.run_id) AS fixes_applied_count
        FROM profiling_runs pr
        LEFT JOIN dataset_profiles orig ON orig.run_id = pr.run_id AND orig.dataset_version = 'original'
        LEFT JOIN dataset_profiles clean ON clean.run_id = pr.run_id AND clean.dataset_version = 'cleaned'
        ORDER BY pr.run_timestamp DESC
    """
    try:
        with connection_scope(db_path) as conn:
            df = pd.read_sql_query(query, conn)
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to load run history: {exc}") from exc

    if not df.empty:
        df["score_improvement"] = (df["cleaned_score"] - df["original_score"]).round(2)
        df["cleaning_applied"] = df["cleaning_applied"].astype(bool)
    return df


def get_run_detail(db_path: Path, run_id: str) -> dict[str, pd.DataFrame]:
    """Load every stored table's rows for a single run.

    Args:
        db_path: Path to the .db file.
        run_id: The run to load.

    Returns:
        A dict of DataFrames keyed by table name: "run", "column_profiles",
        "issues", "scores", "audit_log", "applied_fixes".
    """
    tables = {
        "run": "SELECT * FROM profiling_runs WHERE run_id = ?",
        "column_profiles": "SELECT * FROM column_profiles WHERE run_id = ?",
        "issues": "SELECT * FROM data_quality_issues WHERE run_id = ?",
        "scores": "SELECT * FROM quality_scores WHERE run_id = ?",
        "audit_log": "SELECT * FROM cleaning_audit_log WHERE run_id = ?",
        "applied_fixes": "SELECT * FROM applied_fixes WHERE run_id = ?",
    }
    try:
        with connection_scope(db_path) as conn:
            return {name: pd.read_sql_query(query, conn, params=(run_id,)) for name, query in tables.items()}
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to load details for run '{run_id}': {exc}") from exc
