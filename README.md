# CIP to NOC Evaluation Project

## Purpose

This project evaluates how well Canadian education programs (CIP) align with Canadian occupations (NOC) as **real entry pathways**.

The core question is:

**Does this CIP reasonably prepare someone to enter this NOC occupation?**

The output is a structured model judgment per CIP↔NOC pair, persisted to BigQuery for analysis, quality monitoring, and iterative prompt/model improvement.

## Project scope

The repository is designed for large-batch inference over pre-generated CIP/NOC candidate pairs.

It supports:

- prompt-versioned evaluations (so runs are comparable over time)
- deterministic sharding for parallel processing
- resumable runs (skip rows already written for a run)
- structured result storage for downstream analytics

## Conceptual model

The project treats alignment as a **pathway problem**, not a generic similarity problem.

That means decisions should prioritize:

- NOC employment requirements
- whether the CIP explicitly prepares for that professional path
- credential/licensing/apprenticeship requirements
- domain mismatch vs domain continuity

This avoids false positives from broad transferable skills.

## Infrastructure overview

### Data plane: BigQuery

BigQuery is the system of record for:

- input candidate pairs
- canonical CIP and NOC text
- run metadata
- model outputs

At a high level:

- canonical dataset stores reference CIP/NOC descriptions
- ops dataset stores pairs, run registry, and results

### Compute plane: Cloud Run Jobs

Workers are containerized Python tasks executed as Cloud Run Jobs.

- each task processes a shard (`shard_index` of `n_shards`)
- tasks are stateless and horizontally parallel
- task count/parallelism controls throughput

### Model plane: Ollama-compatible endpoint

Inference calls go to an Ollama API (chat or generate endpoint), typically hosted on Cloud Run (GPU-backed in this environment).

The worker includes:

- OIDC token auth for secure service invocation
- retries/backoff for transient HTTP failures
- JSON parsing/salvage behavior for imperfect model responses

### Delivery plane: Container + deployment script

- Docker image packages worker runtime, config, and prompts
- deploy script builds/pushes image and executes Cloud Run Job with run parameters

## End-to-end strategy

The strategy is to make evaluation quality and throughput both tunable.

1. **Define prompt objective clearly**
   - keep labels and evidence format explicit
   - enforce conservative pathway logic
2. **Run controlled batch evaluations**
   - fixed prompt version
   - fixed CIP/NOC level filters
   - fixed shard and pair caps
3. **Measure output quality in BigQuery**
   - label/evidence distributions
   - missing/invalid outputs
   - latency and failure patterns
4. **Iterate prompt/model config**
   - compare runs by `run_id` and `prompt_version`
   - tighten prompt constraints to reduce false positives
5. **Scale once behavior is stable**
   - increase task count and pair volumes
   - preserve deterministic, auditable run metadata

## General operating principles

- **Reproducibility**: Every run is registered and versioned.
- **Determinism**: Pair selection and shard assignment are hash-based.
- **Resilience**: Retries and per-row error capture prevent silent loss.
- **Auditability**: Raw model response JSON is retained with structured fields.
- **Separation of concerns**: Query building, prompt rendering, model I/O, and writes are isolated in separate modules.

## Repository role in the broader CIP→NOC effort

This repo is the **execution engine** for inference and result capture.

It is not intended to be the source of truth for:

- canonical data modeling decisions
- downstream analytics/reporting surfaces
- policy interpretation of classification standards

Instead, it operationalizes repeatable model evaluation runs so those downstream activities can rely on stable, versioned outputs.

## Current limitations

- prompt set contains legacy and current formats; prompt/output schema alignment must be managed intentionally
- tests are minimal today
- BigQuery schema lifecycle management is external to this repository

## Related files

- `file_descriptions.md`: implementation-level, developer-oriented file-by-file details
- `configs/dev.yaml`: environment-specific project/table/model settings
- `prompts/`: prompt versions used for evaluation strategy experiments
