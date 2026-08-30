# Data Dictionary — SQLite Schema

Full column-level reference for `database/data_quality.db`. Schema source of truth
is `sql/create_tables.sql`; this document explains each field's meaning and origin.

`dataset_version` appears on most tables and is always `'original'` or `'cleaned'`.

## profiling_runs

One row per saved run (one uploaded file, one "Save Run to Database" click).

| Column | Type | Meaning |
|---|---|---|
| `run_id` | TEXT (PK) | UUID identifying this run. Reused across a cleaning session so re-saving without changes is rejected as a duplicate. |
| `dataset_name` | TEXT | Uploaded file name. |
| `file_type` | TEXT | `"csv"` or `"excel"`. |
| `selected_sheet` | TEXT (nullable) | Excel sheet name, when applicable. |
| `run_timestamp` | TEXT | ISO-8601 UTC timestamp when the run was saved. |
| `original_row_count` / `original_column_count` | INTEGER | Shape of the uploaded dataset. |
| `cleaned_row_count` / `cleaned_column_count` | INTEGER (nullable) | Shape after cleaning, if any fixes were applied. |
| `cleaning_applied` | INTEGER (0/1) | Whether cleaning was performed before saving. |
| `schema_version` | TEXT | The `SCHEMA_VERSION` constant this row was written with. |

## dataset_profiles

One row per `(run_id, dataset_version)` — dataset-level stats, mirroring the
Phase 1/2 overview, plus the overall quality score for that version.

| Column | Meaning |
|---|---|
| `row_count`, `column_count`, `total_cells` | Dataset shape. |
| `duplicate_row_count` | Count of `exact_duplicate_row` issues. |
| `empty_row_count`, `empty_column_count` | Count of `empty_row` / `empty_column` issues. |
| `total_missing_count`, `missing_pct` | Combined count of null, blank, whitespace-only, and placeholder cells. |
| `overall_score` | The weighted overall quality score (0-100) for this dataset version. |

## column_profiles

One row per `(run_id, dataset_version, column)` — Phase 2's `ColumnProfile`
fields: `column_name`, `normalized_name`, `pandas_dtype`, `logical_type`,
`confidence`, `non_null_count`, `missing_count`, `missing_pct`, `unique_count`,
`unique_ratio`, `outlier_count` (nullable; IQR-method count for numeric columns only).

> `outlier_count` was added in schema version 1.1.0 via an additive, idempotent
> migration (`database.py:_apply_migrations`) rather than a schema recreation -- a
> database created under 1.0.0 is upgraded in place the next time `init_db()` runs,
> with no data loss.

> Note: the stored `logical_type` is always the *inferred* type (see
> `docs/architecture.md`'s "Human-in-the-loop type override" section). A user's
> session-scoped type override (Column Profiling page) is not persisted to this
> table -- it only affects which issues get detected and saved in
> `data_quality_issues` for that run, not the profile record itself.

## data_quality_issues

One row per detected `Issue` (Phase 3), tagged with `dataset_version` so the same
run can store both the original and post-cleaning issue lists. Columns match
`src/models.py`'s `Issue` dataclass one-to-one: `issue_id`, `column_name`,
`row_index` (nullable — absent for column/dataset-level findings),
`issue_category`, `issue_type`, `severity`, `confidence`, `current_value`,
`suggested_value`, `recommended_action`, `safe_to_auto_fix`, `rule_name`,
`detected_at`.

## cleaning_audit_log

One row per applied change (Phase 4), matching `AuditLogEntry`: `timestamp`,
`dataset_name`, `row_index`, `column_name`, `original_value`, `new_value`,
`cleaning_action`, `rule_name`, `reason`, `user_approved`, `confidence`. Only
populated when cleaning was performed.

## applied_fixes

A rollup of the audit log: one row per fix type that was applied in a run
(`fix_id`, `fix_title`, `affected_count`), rather than one row per changed cell.

## quality_scores

One row per `(run_id, dataset_version, component_name)` — the full, explainable
breakdown behind `dataset_profiles.overall_score`: `component_score`,
`issue_count`, `denominator`, `penalty`, `component_weight`,
`weighted_contribution`, `calculation_explanation` (a human-readable sentence
generated from the same numbers), `calculated_at`.

## Relationships

Every table except `profiling_runs` and `schema_info` has a `run_id` foreign key
to `profiling_runs(run_id)` with `ON DELETE CASCADE`, enforced via
`PRAGMA foreign_keys = ON` on every connection (`src/database.py:get_connection`).
Deleting a `profiling_runs` row (not yet exposed in the UI — see "Clear Current
Unsaved Session", which only clears in-memory Streamlit state, never the database)
would cascade-delete all of that run's child rows.

## What is never stored here

The full uploaded or cleaned dataset is never written to this database — only
profiles, issue records, audit log entries, and scores. Row-level fields
(`row_index`, `current_value`, `original_value`, `new_value`) reference the source
file's row position and value *at detection/cleaning time*, which is enough to
explain and audit a finding without duplicating the dataset itself.
