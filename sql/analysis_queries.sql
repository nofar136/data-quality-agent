-- Generic AI Data Quality Agent -- example analysis queries (Phase 5)
--
-- Reference queries for exploring database/data_quality.db directly (e.g.
-- via the sqlite3 CLI or DB Browser for SQLite), demonstrating the schema's
-- normalized structure. The app itself uses equivalent logic in
-- src/database.py for the in-app run history and comparison views.

-- 1. Latest profiling runs
SELECT run_id, dataset_name, run_timestamp, cleaning_applied,
       original_row_count, original_column_count
FROM profiling_runs
ORDER BY run_timestamp DESC
LIMIT 20;

-- 2. Overall score by run (original vs cleaned, side by side)
SELECT pr.run_id, pr.dataset_name, pr.run_timestamp,
       orig.overall_score AS original_score,
       clean.overall_score AS cleaned_score
FROM profiling_runs pr
LEFT JOIN dataset_profiles orig ON orig.run_id = pr.run_id AND orig.dataset_version = 'original'
LEFT JOIN dataset_profiles clean ON clean.run_id = pr.run_id AND clean.dataset_version = 'cleaned'
ORDER BY pr.run_timestamp DESC;

-- 3. Original versus cleaned score, with the improvement
SELECT pr.run_id, pr.dataset_name,
       orig.overall_score AS original_score,
       clean.overall_score AS cleaned_score,
       ROUND(clean.overall_score - orig.overall_score, 2) AS score_improvement
FROM profiling_runs pr
JOIN dataset_profiles orig ON orig.run_id = pr.run_id AND orig.dataset_version = 'original'
JOIN dataset_profiles clean ON clean.run_id = pr.run_id AND clean.dataset_version = 'cleaned'
ORDER BY score_improvement DESC;

-- 4. Issues by category (for a specific run)
SELECT issue_category, COUNT(*) AS issue_count
FROM data_quality_issues
WHERE run_id = :run_id AND dataset_version = 'original'
GROUP BY issue_category
ORDER BY issue_count DESC;

-- 5. Issues by severity (for a specific run)
SELECT severity, COUNT(*) AS issue_count
FROM data_quality_issues
WHERE run_id = :run_id AND dataset_version = 'original'
GROUP BY severity
ORDER BY CASE severity WHEN 'Critical' THEN 1 WHEN 'High' THEN 2 WHEN 'Medium' THEN 3 ELSE 4 END;

-- 6. Columns with the most issues (across all runs for one dataset)
SELECT column_name, COUNT(*) AS issue_count
FROM data_quality_issues q
JOIN profiling_runs pr ON pr.run_id = q.run_id
WHERE pr.dataset_name = :dataset_name AND column_name IS NOT NULL
GROUP BY column_name
ORDER BY issue_count DESC
LIMIT 10;

-- 7. Number of fixes applied by run
SELECT pr.run_id, pr.dataset_name, COUNT(af.id) AS fixes_applied,
       SUM(af.affected_count) AS total_changes
FROM profiling_runs pr
LEFT JOIN applied_fixes af ON af.run_id = pr.run_id
GROUP BY pr.run_id
ORDER BY pr.run_timestamp DESC;

-- 8. Average score across datasets (original scores only)
SELECT pr.dataset_name, ROUND(AVG(dp.overall_score), 2) AS avg_original_score, COUNT(*) AS run_count
FROM profiling_runs pr
JOIN dataset_profiles dp ON dp.run_id = pr.run_id AND dp.dataset_version = 'original'
GROUP BY pr.dataset_name
ORDER BY avg_original_score DESC;

-- 9. Runs with the greatest improvement after cleaning
SELECT pr.run_id, pr.dataset_name, pr.run_timestamp,
       orig.overall_score AS original_score,
       clean.overall_score AS cleaned_score,
       ROUND(clean.overall_score - orig.overall_score, 2) AS score_improvement
FROM profiling_runs pr
JOIN dataset_profiles orig ON orig.run_id = pr.run_id AND orig.dataset_version = 'original'
JOIN dataset_profiles clean ON clean.run_id = pr.run_id AND clean.dataset_version = 'cleaned'
WHERE pr.cleaning_applied = 1
ORDER BY score_improvement DESC
LIMIT 10;

-- 10. Component score breakdown for a specific run and dataset version
SELECT component_name, component_score, issue_count, denominator, penalty,
       component_weight, weighted_contribution, calculation_explanation
FROM quality_scores
WHERE run_id = :run_id AND dataset_version = :dataset_version
ORDER BY component_weight DESC;
