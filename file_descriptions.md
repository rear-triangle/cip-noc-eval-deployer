# CIP-NOC Eval Deployer

Batch evaluation pipeline for scoring whether a Canadian CIP program is an educational pathway into a Canadian NOC occupation using an LLM, with BigQuery as the input/output system of record.

## What this repository does

This repo runs large-scale CIP↔NOC pair evaluations by:

1. Selecting CIP/NOC candidate pairs from BigQuery.
2. Joining canonical CIP and NOC text fields.
3. Rendering a prompt template per pair.
4. Calling an Ollama-compatible LLM endpoint (typically Cloud Run-hosted).
5. Writing structured results back to BigQuery.

The system is designed for sharded parallel execution on Cloud Run Jobs (or locally) and reproducible prompt-versioned runs.

## High-level architecture

- **Data source**: BigQuery `ops.pairs` + canonical CIP/NOC tables.
- **Orchestration metadata**: BigQuery `ops.run_registry`.
- **Worker runtime**: Python async worker (`src/worker.py`).
- **Model API**: Ollama `/api/chat` (default) or `/api/generate`.
- **Result sink**: BigQuery `ops.results`.

### Execution flow

1. Create a run ID and registry row (`src/run_eval.py`).
2. Start N workers (shards), each with `shard_index`.
3. Each worker:
   - Builds a deterministic SQL query (`src/bq_queries.py`) to fetch its shard of unprocessed pairs.
   - Renders prompt placeholders (`src/prompt_loader.py`).
   - Calls model with retries/OIDC auth (`src/llm_client.py`).
   - Inserts deduplicated rows into BigQuery (`src/result_writer.py`).

## Repository layout

- `src/worker.py`: main async worker entrypoint (CLI/env-driven).
- `src/run_eval.py`: run bootstrapper; inserts `run_registry` row and prints per-shard worker commands.
- `src/bq_queries.py`: SQL builder for selecting/joining pending pairs.
- `src/llm_client.py`: resilient Ollama client with OIDC token minting and response salvage logic.
- `src/prompt_loader.py`: prompt loading + token substitution (`{{...}}`).
- `src/result_writer.py`: batched BigQuery writer with row IDs (`run_id:pair_id`) for idempotency.
- `configs/dev.yaml`: project/dataset/table/LLM defaults.
- `prompts/*.txt`: versioned prompt templates.
- `Dockerfile`: worker container image.
- `deploy.sh`: build/push image and execute Cloud Run Job with env vars.
- `bq_monitor.sh`: post-run QA/monitor queries in BigQuery.
- `cip-noc-worker.yaml`: Cloud Run Job manifest example.
- `ollama-gcs-llama3-gpu.yaml`: Cloud Run Service manifest for GPU Ollama backend.
- `tests/`: currently empty.

## Requirements

- Python 3.11+
- `gcloud` CLI (for Cloud Run / BigQuery ops)
- Access to target GCP project/datasets/tables
- IAM to:
  - Query/insert in BigQuery datasets
  - Invoke target LLM service (Cloud Run `run.invoker`)

Python dependencies (`requirements.txt`):

- `google-cloud-bigquery==3.25.0`
- `google-auth==2.33.0`
- `aiohttp==3.13.3`
- `PyYAML==6.0.2`

## BigQuery tables expected by config

Defined in `configs/dev.yaml`:

- Canonical dataset: `llm_cip_noc_canonical`
  - `cip_canonical`
  - `noc_canonical`
- Ops dataset: `llm_cip_noc_ops`
  - `cip_noc_pairs`
  - `run_registry`
  - `results`

### Key write behavior

- `run_eval.py` writes one row to `run_registry` with run metadata.
- `worker.py` writes one row per evaluated pair to `results`.
- Worker query excludes already-completed `(run_id, pair_id)` rows, making reruns resumable.

## Configuration

Primary config file: `configs/dev.yaml`.

Important keys:

- `project_id`, `location`
- `datasets.canonical`, `datasets.ops`
- `tables.*`
- `llm.base_url`, `llm.generate_path`, `llm.model`
- retry/time/concurrency knobs (`timeout_s`, `max_retries`, `concurrency`, `batch_size`)
- warmup/stagger knobs (`stagger_s`, `ready_retries`, `ready_sleep_s`)
- generation knobs (`temperature`, `top_p`, `num_predict`, `stop_csv`)
- default sharding (`run_defaults.n_shards`)

## Prompt system

- Prompts are plain text templates in `prompts/`.
- Substitution is direct string replacement for tokens like `{{cip_title}}`, `{{noc_main_duties}}`, `{{prompt_version}}`.
- `--prompt-version` is tracked in BigQuery and should match the template version in the filename.

Note: newer code expects fields like `label` + `evidence_grade` in model output. Older prompts in this repo still reference `confidence`; avoid those for current pipeline unless you also adjust parser/schema expectations.

## Local development setup

Install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Authenticate:

```bash
gcloud auth application-default login
```

If invoking protected Cloud Run LLM from local machine, either:

- rely on ADC minting ID tokens automatically, or
- set `MANUAL_ID_TOKEN`:

```bash
export MANUAL_ID_TOKEN="$(gcloud auth print-identity-token --audiences='https://YOUR-LLM-URL')"
```

## Running a full evaluation

### Step 1: Create run metadata and get shard commands

```bash
python -m src.run_eval \
  --config configs/dev.yaml \
  --prompt-path prompts/v010_prompt.txt \
  --prompt-version v010 \
  --cip-levels 3 \
  --noc-levels 5 \
  --n-shards 10 \
  --max-pairs 1000 \
  --notes "local test"
```

This prints a generated `run_id` and one `python -m src.worker ... --shard-index N` command per shard.

### Step 2: Start worker(s)

Run one shard locally:

```bash
python -m src.worker \
  --config configs/dev.yaml \
  --run-id <RUN_ID> \
  --prompt-path prompts/v010_prompt.txt \
  --prompt-version v010 \
  --cip-levels 3 \
  --noc-levels 5 \
  --n-shards 10 \
  --shard-index 0 \
  --max-pairs 1000
```

Or launch all shard commands in parallel from the output of step 1.

## Cloud Run deployment/execution

`deploy.sh` does:

1. Build/push Docker image to Artifact Registry.
2. Update an existing Cloud Run Job image.
3. Execute the job with run-specific env vars (`RUN_ID`, prompt info, filters).

Run it:

```bash
./deploy.sh
```

Before running, verify constants inside `deploy.sh` (`PROJECT_ID`, `REGION`, `JOB`, prompt version/path, filter levels, max pairs).

## Monitoring and quality checks

`bq_monitor.sh` runs post-run checks directly in BigQuery:

- prompt version distribution
- missing outputs
- label distribution
- evidence grade distribution
- latency percentiles
- sample suspicious/high-latency rows

Usage:

```bash
TABLE=lmic-dev-datahub.llm_cip_noc_ops.results RUN_ID=<RUN_ID> ./bq_monitor.sh
```

If `RUN_ID` is omitted, script can resolve the latest run (optionally by `PROMPT_VERSION`).

## Container behavior

`Dockerfile` builds a worker image and sets:

- `PYTHONPATH=/app`
- `ENTRYPOINT ["python", "-m", "src.worker"]`

So container startup expects worker args via env vars or CLI args supplied by Cloud Run Job.

## Runtime inputs and env vars

`src/worker.py` accepts both CLI flags and env vars:

- `--config` / `CONFIG_PATH`
- `--run-id` / `RUN_ID`
- `--prompt-path` / `PROMPT_PATH`
- `--prompt-version` / `PROMPT_VERSION`
- `--cip-levels` / `CIP_LEVELS`
- `--noc-levels` / `NOC_LEVELS`
- `--max-pairs` / `MAX_PAIRS`
- `--n-shards` / `CLOUD_RUN_TASK_COUNT` fallback
- `--shard-index` / `CLOUD_RUN_TASK_INDEX` fallback

LLM/auth-related env vars used by `src/llm_client.py` include:

- `MANUAL_ID_TOKEN`
- `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`
- `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_NUM_PREDICT`, `LLM_STOP`
- `LLM_TOKEN_REFRESH_SKEW_S`, `LLM_ALLOW_MANUAL_TOKEN`
- `LLM_SEED` (optional reproducibility)

## Operational notes

- Sharding is deterministic via `FARM_FINGERPRINT(pair_key)`.
- `max_pairs` is applied before sharding using a stable hash order.
- Inserts use deterministic `row_ids` (`run_id:pair_id`) to reduce duplicate writes.
- Worker stores raw model JSON in `response_json` for audit/debug.
- On per-row inference failure, worker inserts an error payload row instead of dropping the item.

## Known gaps

- No automated tests are currently present in `tests/`.
- No schema migration or IaC for BigQuery tables is included in this repo.
- Prompt inventory includes older formats; align prompt output keys with current parser before production runs.

## Quick onboarding checklist for new developers

1. Read `configs/dev.yaml` and confirm project/dataset/table names.
2. Verify IAM and BigQuery access.
3. Pick a current prompt (recommended: `prompts/v010_prompt.txt`).
4. Run `src.run_eval` with small `--max-pairs`.
5. Execute one worker shard locally to validate LLM auth + output format.
6. Scale to Cloud Run Job and monitor with `bq_monitor.sh`.
