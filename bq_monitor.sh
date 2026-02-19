#!/usr/bin/env bash
set -euo pipefail

# Required
: "${TABLE:?TABLE must be set (e.g., lmic-dev-datahub.llm_cip_noc_ops.results)}"

# Optional inputs:
# - RUN_ID: if provided, use it.
# - PROMPT_VERSION: if RUN_ID not provided, pick latest run_id matching this prompt_version.
# - If neither RUN_ID nor PROMPT_VERSION provided, pick latest run_id overall.

if [[ -z "${RUN_ID:-}" ]]; then
  if [[ -n "${PROMPT_VERSION:-}" ]]; then
    echo "RUN_ID not set; selecting latest run_id for PROMPT_VERSION=${PROMPT_VERSION} from TABLE=${TABLE} ..."
    RUN_ID="$(
      bq query --use_legacy_sql=false --format=csv --quiet "
        SELECT run_id
        FROM \`${TABLE}\`
        WHERE prompt_version = '${PROMPT_VERSION}'
        GROUP BY run_id
        ORDER BY MAX(created_at) DESC
        LIMIT 1;
      " | tail -n 1
    )"
  else
    echo "RUN_ID not set; selecting latest run_id overall from TABLE=${TABLE} ..."
    RUN_ID="$(
      bq query --use_legacy_sql=false --format=csv --quiet "
        SELECT run_id
        FROM \`${TABLE}\`
        GROUP BY run_id
        ORDER BY MAX(created_at) DESC
        LIMIT 1;
      " | tail -n 1
    )"
  fi
fi

if [[ -z "${RUN_ID}" || "${RUN_ID}" == "run_id" ]]; then
  echo "ERROR: Could not resolve RUN_ID from TABLE=${TABLE} (check TABLE, PROMPT_VERSION, or permissions)."
  exit 1
fi

echo "RUN_ID=${RUN_ID}"
echo "TABLE=${TABLE}"
echo "PROMPT_VERSION=${PROMPT_VERSION:-<not set>}"
echo

# Helper: resolve evidence_grade from either a real column or response_json (transition-safe).
# NOTE: We can't define SQL functions in bq CLI easily, so we inline COALESCE(...) where needed.

# 1) Confirm prompt_version distribution (should match expected prompt)
bq query --use_legacy_sql=false "
SELECT prompt_version, COUNT(*) AS n
FROM \`${TABLE}\`
WHERE run_id = \"${RUN_ID}\"
GROUP BY prompt_version
ORDER BY n DESC;
"

# 2) Sanity: outputs present (evidence_grade replaces confidence)
bq query --use_legacy_sql=false "
SELECT COUNT(*) AS rows_missing_outputs
FROM \`${TABLE}\`
WHERE run_id = \"${RUN_ID}\"
  AND (
    label IS NULL
    OR rationale IS NULL
    OR model_version IS NULL
    OR COALESCE(CAST(evidence_grade AS STRING), JSON_VALUE(response_json, '\$.evidence_grade')) IS NULL
  );
"

# 3) Sanity: unique inputs (hashes)
bq query --use_legacy_sql=false "
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT input_payload_hash) AS distinct_input_hashes
FROM \`${TABLE}\`
WHERE run_id = \"${RUN_ID}\";
"

# 4) Label distribution
bq query --use_legacy_sql=false "
SELECT label, COUNT(*) AS n
FROM \`${TABLE}\`
WHERE run_id = \"${RUN_ID}\"
GROUP BY label
ORDER BY n DESC;
"

# 5) Evidence grade distribution (replaces confidence buckets)
bq query --use_legacy_sql=false "
SELECT
  COALESCE(CAST(evidence_grade AS STRING), JSON_VALUE(response_json, '\$.evidence_grade')) AS evidence_grade,
  COUNT(*) AS n
FROM \`${TABLE}\`
WHERE run_id = \"${RUN_ID}\"
GROUP BY evidence_grade
ORDER BY n DESC;
"

# 6) Extra keys check (RE2-safe): extract keys then find unexpected
bq query --use_legacy_sql=false "
WITH keys_per_row AS (
  SELECT
    pair_id,
    ARRAY(
      SELECT DISTINCT k
      FROM UNNEST(REGEXP_EXTRACT_ALL(response_json, r'\"([a-zA-Z0-9_]+)\"\\s*:')) AS k
    ) AS keys
  FROM \`${TABLE}\`
  WHERE run_id = \"${RUN_ID}\"
),
unexpected AS (
  SELECT
    pair_id,
    k AS unexpected_key
  FROM keys_per_row, UNNEST(keys) AS k
  WHERE k NOT IN (
    'label','evidence_grade','rationale','model_version','prompt_version',
    'parse_error','raw_response','ollama_meta','salvaged','error'
  )
)
SELECT
  COUNT(DISTINCT pair_id) AS rows_with_extra_keys,
  ARRAY_AGG(DISTINCT unexpected_key ORDER BY unexpected_key LIMIT 50) AS sample_unexpected_keys
FROM unexpected;
"

# 7) Latency stats (overall + percentiles)
bq query --use_legacy_sql=false "
SELECT
  COUNT(*) AS n,
  COUNTIF(latency_ms IS NULL) AS missing_latency,
  MIN(latency_ms) AS min_ms,
  APPROX_QUANTILES(latency_ms, 100)[OFFSET(50)] AS p50_ms,
  APPROX_QUANTILES(latency_ms, 100)[OFFSET(90)] AS p90_ms,
  APPROX_QUANTILES(latency_ms, 100)[OFFSET(95)] AS p95_ms,
  APPROX_QUANTILES(latency_ms, 100)[OFFSET(99)] AS p99_ms,
  MAX(latency_ms) AS max_ms,
  AVG(latency_ms) AS avg_ms
FROM \`${TABLE}\`
WHERE run_id = \"${RUN_ID}\";
"

# 8) Latency by label
bq query --use_legacy_sql=false "
SELECT
  label,
  COUNT(*) AS n,
  APPROX_QUANTILES(latency_ms, 100)[OFFSET(50)] AS p50_ms,
  APPROX_QUANTILES(latency_ms, 100)[OFFSET(90)] AS p90_ms,
  APPROX_QUANTILES(latency_ms, 100)[OFFSET(95)] AS p95_ms,
  APPROX_QUANTILES(latency_ms, 100)[OFFSET(99)] AS p99_ms,
  AVG(latency_ms) AS avg_ms
FROM \`${TABLE}\`
WHERE run_id = \"${RUN_ID}\"
  AND latency_ms IS NOT NULL
GROUP BY label
ORDER BY p95_ms DESC;
"

# 9) Worst latency examples
bq query --use_legacy_sql=false "
SELECT
  pair_id,
  latency_ms,
  label,
  COALESCE(CAST(evidence_grade AS STRING), JSON_VALUE(response_json, '\$.evidence_grade')) AS evidence_grade,
  cip_code,
  cip_title,
  noc_code,
  noc_title,
  SUBSTR(rationale, 1, 200) AS rationale_200
FROM \`${TABLE}\`
WHERE run_id = \"${RUN_ID}\"
  AND latency_ms IS NOT NULL
ORDER BY latency_ms DESC
LIMIT 25;
"

# 10) Random sample for manual rationale inspection
bq query --use_legacy_sql=false "
SELECT
  label,
  COALESCE(CAST(evidence_grade AS STRING), JSON_VALUE(response_json, '\$.evidence_grade')) AS evidence_grade,
  cip_code,
  cip_title,
  noc_code,
  noc_title,
  rationale
FROM \`${TABLE}\`
WHERE run_id = \"${RUN_ID}\"
ORDER BY RAND()
LIMIT 30;
"

# 11) Top-grade examples per label (replaces high-confidence examples)
# Assumes evidence_grade uses letters where A is strongest; adjust ordering if you invert.
bq query --use_legacy_sql=false "
WITH ranked AS (
  SELECT
    label,
    COALESCE(CAST(evidence_grade AS STRING), JSON_VALUE(response_json, '\$.evidence_grade')) AS evidence_grade,
    cip_code, cip_title, noc_code, noc_title, rationale,
    ROW_NUMBER() OVER (
      PARTITION BY label
      ORDER BY
        CASE COALESCE(CAST(evidence_grade AS STRING), JSON_VALUE(response_json, '\$.evidence_grade'))
          WHEN 'A' THEN 1
          WHEN 'B' THEN 2
          WHEN 'C' THEN 3
          WHEN 'D' THEN 4
          WHEN 'F' THEN 5
          ELSE 6
        END ASC
    ) AS rn
  FROM \`${TABLE}\`
  WHERE run_id = \"${RUN_ID}\"
)
SELECT *
FROM ranked
WHERE rn <= 15
ORDER BY label, rn;
"

# 12) Auto-flag suspicious YES (domain mismatch heuristics)
# Note: Sorting by evidence strength now uses grade order (A strongest).
bq query --use_legacy_sql=false "
SELECT
  label,
  COALESCE(CAST(evidence_grade AS STRING), JSON_VALUE(response_json, '\$.evidence_grade')) AS evidence_grade,
  cip_code,
  cip_title,
  noc_code,
  noc_title,
  rationale
FROM \`${TABLE}\`
WHERE run_id = \"${RUN_ID}\"
  AND label = 'YES'
  AND (
    (REGEXP_CONTAINS(LOWER(cip_title), r'engineering|chemistry|physics|mathematics|statistics|computer')
      AND REGEXP_CONTAINS(LOWER(noc_title), r'translator|interpreter|archivist|carpenter|artist|writer|teacher|sales|deckhand|crew|labourer'))
    OR
    (REGEXP_CONTAINS(LOWER(cip_title), r'criminal justice|police|history|literature|philosophy|religion|language')
      AND REGEXP_CONTAINS(LOWER(noc_title), r'electrician|plumber|engineer|nurse|dentist|lawyer|carpenter'))
  )
ORDER BY
  CASE COALESCE(CAST(evidence_grade AS STRING), JSON_VALUE(response_json, '\$.evidence_grade'))
    WHEN 'A' THEN 1
    WHEN 'B' THEN 2
    WHEN 'C' THEN 3
    WHEN 'D' THEN 4
    WHEN 'F' THEN 5
    ELSE 6
  END ASC
LIMIT 100;
"