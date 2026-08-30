-- Generic AI Data Quality Agent -- SQLite schema (Phase 5)
--
-- Safe to run repeatedly: every statement is IF NOT EXISTS, so re-running
-- this file against an existing database is a no-op.
--
-- The full uploaded/cleaned dataset is never stored here by default --
-- only profiles, issue records, audit log entries, applied-fix summaries,
-- and quality scores. Row-level fields (row_index, current_value,
-- original_value, new_value) reference the source file's row position and
-- values at detection/cleaning time, not a persisted copy of the dataset.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_info (
    schema_version TEXT PRIMARY KEY,
    applied_at     TEXT NOT NULL
);

-- One row per profiling/cleaning run (one uploaded file, one session).
CREATE TABLE IF NOT EXISTS profiling_runs (
    run_id                 TEXT PRIMARY KEY,
    dataset_name           TEXT NOT NULL,
    file_type              TEXT NOT NULL,
    selected_sheet         TEXT,
    run_timestamp          TEXT NOT NULL,
    original_row_count     INTEGER NOT NULL,
    original_column_count  INTEGER NOT NULL,
    cleaned_row_count      INTEGER,
    cleaned_column_count   INTEGER,
    cleaning_applied       INTEGER NOT NULL DEFAULT 0 CHECK (cleaning_applied IN (0, 1)),
    schema_version         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profiling_runs_dataset_name ON profiling_runs(dataset_name);
CREATE INDEX IF NOT EXISTS idx_profiling_runs_timestamp ON profiling_runs(run_timestamp);

-- One row per (run, dataset_version) -- dataset_version is 'original' or
-- 'cleaned'. Dataset-level profile stats, mirroring Phase 1/2's overview.
CREATE TABLE IF NOT EXISTS dataset_profiles (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               TEXT NOT NULL REFERENCES profiling_runs(run_id) ON DELETE CASCADE,
    dataset_version      TEXT NOT NULL CHECK (dataset_version IN ('original', 'cleaned')),
    row_count            INTEGER NOT NULL,
    column_count         INTEGER NOT NULL,
    total_cells          INTEGER NOT NULL,
    duplicate_row_count  INTEGER NOT NULL,
    empty_row_count      INTEGER NOT NULL,
    empty_column_count   INTEGER NOT NULL,
    total_missing_count  INTEGER NOT NULL,
    missing_pct          REAL NOT NULL,
    overall_score        REAL NOT NULL,
    UNIQUE(run_id, dataset_version)
);
CREATE INDEX IF NOT EXISTS idx_dataset_profiles_run ON dataset_profiles(run_id);

-- One row per (run, dataset_version, column) -- Phase 2's column profile.
CREATE TABLE IF NOT EXISTS column_profiles (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL REFERENCES profiling_runs(run_id) ON DELETE CASCADE,
    dataset_version  TEXT NOT NULL CHECK (dataset_version IN ('original', 'cleaned')),
    column_name      TEXT NOT NULL,
    normalized_name  TEXT NOT NULL,
    pandas_dtype     TEXT NOT NULL,
    logical_type     TEXT NOT NULL,
    confidence       REAL NOT NULL,
    non_null_count   INTEGER NOT NULL,
    missing_count    INTEGER NOT NULL,
    missing_pct      REAL NOT NULL,
    unique_count     INTEGER NOT NULL,
    unique_ratio     REAL NOT NULL,
    outlier_count    INTEGER  -- NULL for non-numeric columns; added in schema 1.1.0 (see database.py:_apply_migrations)
);
CREATE INDEX IF NOT EXISTS idx_column_profiles_run ON column_profiles(run_id);

-- One row per detected issue (Phase 3), tagged by which dataset_version it
-- was found in.
CREATE TABLE IF NOT EXISTS data_quality_issues (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES profiling_runs(run_id) ON DELETE CASCADE,
    dataset_version     TEXT NOT NULL CHECK (dataset_version IN ('original', 'cleaned')),
    issue_id            TEXT NOT NULL,
    column_name         TEXT,
    row_index           INTEGER,
    issue_category      TEXT NOT NULL,
    issue_type          TEXT NOT NULL,
    severity            TEXT NOT NULL CHECK (severity IN ('Low', 'Medium', 'High', 'Critical')),
    confidence          REAL NOT NULL,
    current_value       TEXT,
    suggested_value     TEXT,
    recommended_action  TEXT NOT NULL,
    safe_to_auto_fix    INTEGER NOT NULL CHECK (safe_to_auto_fix IN (0, 1)),
    rule_name           TEXT NOT NULL,
    detected_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_issues_run ON data_quality_issues(run_id);
CREATE INDEX IF NOT EXISTS idx_issues_category ON data_quality_issues(issue_category);
CREATE INDEX IF NOT EXISTS idx_issues_severity ON data_quality_issues(severity);
CREATE INDEX IF NOT EXISTS idx_issues_column ON data_quality_issues(column_name);

-- One row per applied cleaning change (Phase 4's audit log).
CREATE TABLE IF NOT EXISTS cleaning_audit_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL REFERENCES profiling_runs(run_id) ON DELETE CASCADE,
    timestamp        TEXT NOT NULL,
    dataset_name     TEXT NOT NULL,
    row_index        INTEGER,
    column_name      TEXT,
    original_value   TEXT,
    new_value        TEXT,
    cleaning_action  TEXT NOT NULL,
    rule_name        TEXT NOT NULL,
    reason           TEXT NOT NULL,
    user_approved    INTEGER NOT NULL CHECK (user_approved IN (0, 1)),
    confidence       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_run ON cleaning_audit_log(run_id);

-- One row per fix type that was applied in a run (a rollup of the audit
-- log, one row per fix_id rather than one row per changed cell).
CREATE TABLE IF NOT EXISTS applied_fixes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL REFERENCES profiling_runs(run_id) ON DELETE CASCADE,
    fix_id           TEXT NOT NULL,
    fix_title        TEXT NOT NULL,
    affected_count   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_applied_fixes_run ON applied_fixes(run_id);

-- One row per (run, dataset_version, component) -- Phase 5's score
-- breakdown. The overall weighted score is stored once per dataset_version
-- on dataset_profiles.overall_score, not duplicated here as a fake
-- "component" row.
CREATE TABLE IF NOT EXISTS quality_scores (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                   TEXT NOT NULL REFERENCES profiling_runs(run_id) ON DELETE CASCADE,
    dataset_version          TEXT NOT NULL CHECK (dataset_version IN ('original', 'cleaned')),
    component_name           TEXT NOT NULL,
    component_score          REAL NOT NULL,
    issue_count              INTEGER NOT NULL,
    denominator              INTEGER NOT NULL,
    penalty                  REAL NOT NULL,
    component_weight         REAL NOT NULL,
    weighted_contribution    REAL NOT NULL,
    calculation_explanation  TEXT NOT NULL,
    calculated_at            TEXT NOT NULL,
    UNIQUE(run_id, dataset_version, component_name)
);
CREATE INDEX IF NOT EXISTS idx_scores_run ON quality_scores(run_id);
