-- =================================================================
-- queries_day3.sql
-- W5D3 SQL Queries on the pipeline_results table
-- Author : Umed
-- Table  : pipeline_results
--          (id, task, model, input_text, output_json, created_at)
-- =================================================================

-- -----------------------------------------------------------------
-- Query 1 (Required):
-- Count total pipeline calls made TODAY, grouped by task
-- Useful to see which NLP task was used most in a session
-- -----------------------------------------------------------------
SELECT
    task,
    COUNT(*) AS total_calls
FROM pipeline_results
WHERE DATE(created_at) = DATE('now')
GROUP BY task
ORDER BY total_calls DESC;


-- -----------------------------------------------------------------
-- Query 2 (Required):
-- Find all sentiment analysis results where output is NEGATIVE
-- Uses json_extract() for proper JSON parsing (SQLite 3.38+)
-- LIKE fallback is commented below for older SQLite versions
-- -----------------------------------------------------------------
SELECT
    id,
    input_text,
    json_extract(output_json, '$[0].label') AS sentiment_label,
    ROUND(json_extract(output_json, '$[0].score'), 4) AS confidence,
    created_at
FROM pipeline_results
WHERE task = 'sentiment-analysis'
  AND json_extract(output_json, '$[0].label') = 'NEGATIVE';

-- LIKE fallback (works on all SQLite versions):
-- SELECT id, input_text, output_json, created_at
-- FROM pipeline_results
-- WHERE task = 'sentiment-analysis'
--   AND output_json LIKE '%NEGATIVE%';


-- -----------------------------------------------------------------
-- Query 3 (Required):
-- List all UNIQUE models used, with the task they were used for
-- Helps track which models were tested across the pipeline
-- -----------------------------------------------------------------
SELECT DISTINCT
    model,
    task
FROM pipeline_results
ORDER BY task, model;


-- -----------------------------------------------------------------
-- Query 4 (Required):
-- Find the LONGEST input text submitted to any pipeline today
-- Uses LENGTH() to measure character count of input_text
-- -----------------------------------------------------------------
SELECT
    id,
    task,
    model,
    LENGTH(input_text)          AS input_length,
    SUBSTR(input_text, 1, 80)   AS input_preview
FROM pipeline_results
WHERE DATE(created_at) = DATE('now')
ORDER BY input_length DESC
LIMIT 1;


-- -----------------------------------------------------------------
-- Query 5 (Custom):
-- Average confidence score per task per model
-- Useful for comparing how "certain" each model is on its outputs.
-- A lower average confidence may indicate the model is struggling
-- with out-of-domain inputs (e.g. Urdu text fed to English model).
-- -----------------------------------------------------------------
SELECT
    task,
    model,
    COUNT(*)  AS total_calls,
    ROUND(AVG(CAST(json_extract(output_json, '$[0].score') AS REAL)), 4)
              AS avg_confidence,
    ROUND(MIN(CAST(json_extract(output_json, '$[0].score') AS REAL)), 4)
              AS min_confidence,
    ROUND(MAX(CAST(json_extract(output_json, '$[0].score') AS REAL)), 4)
              AS max_confidence
FROM pipeline_results
GROUP BY task, model
ORDER BY task, avg_confidence DESC;