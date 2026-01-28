# src/worker.py

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp
import yaml
from google.cloud import bigquery

from src.bq_queries import DatasetRefs, TableRefs, build_inputs_query
from src.llm_client import LLMClient, LLMClientConfig
from src.prompt_loader import load_prompt, render_prompt
from src.result_writer import ResultWriter, ResultsTarget


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def _csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _safe_get(obj: Dict[str, Any], key: str, default=None):
    return obj.get(key, default)


def _build_parser() -> argparse.ArgumentParser:
    """
    Support BOTH CLI flags and env-var fallbacks.

    Env vars supported:
      CONFIG_PATH
      RUN_ID
      PROMPT_PATH
      PROMPT_VERSION
      CIP_LEVELS
      NOC_LEVELS
      MAX_PAIRS
    """
    p = argparse.ArgumentParser()

    p.add_argument("--config", default=_env("CONFIG_PATH"))
    p.add_argument("--run-id", default=_env("RUN_ID"))

    p.add_argument("--prompt-path", default=_env("PROMPT_PATH"))
    p.add_argument("--prompt-version", default=_env("PROMPT_VERSION"))

    p.add_argument("--cip-levels", default=_env("CIP_LEVELS"), help="comma-separated, e.g. 3 or 1,2,3")
    p.add_argument("--noc-levels", default=_env("NOC_LEVELS"), help="comma-separated, e.g. 5 or 4,5")

    # Cloud Run Jobs sets these automatically per task
    p.add_argument(
        "--n-shards",
        type=int,
        default=int(os.getenv("CLOUD_RUN_TASK_COUNT", "1")),
        help="defaults to CLOUD_RUN_TASK_COUNT if set",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=int(os.getenv("CLOUD_RUN_TASK_INDEX", "0")),
        help="defaults to CLOUD_RUN_TASK_INDEX if set",
    )

    p.add_argument("--max-pairs", type=int, default=int(_env("MAX_PAIRS", "0") or "0"), help="0 means no cap")
    return p


def _validate_args(args: argparse.Namespace) -> List[str]:
    missing = []
    if not args.config:
        missing.append("--config / CONFIG_PATH")
    if not args.run_id:
        missing.append("--run-id / RUN_ID")
    if not args.prompt_path:
        missing.append("--prompt-path / PROMPT_PATH")
    if not args.prompt_version:
        missing.append("--prompt-version / PROMPT_VERSION")
    if not args.cip_levels:
        missing.append("--cip-levels / CIP_LEVELS")
    if not args.noc_levels:
        missing.append("--noc-levels / NOC_LEVELS")
    return missing


async def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    missing = _validate_args(args)

    # ALWAYS log something early (even if we will exit).
    print(
        json.dumps(
            {
                "msg": "worker starting",
                "run_id": args.run_id,
                "config": args.config,
                "prompt_path": args.prompt_path,
                "prompt_version": args.prompt_version,
                "cip_levels": args.cip_levels,
                "noc_levels": args.noc_levels,
                "n_shards": args.n_shards,
                "shard_index": args.shard_index,
                "max_pairs": args.max_pairs,
                "task_index_env": os.getenv("CLOUD_RUN_TASK_INDEX"),
                "task_count_env": os.getenv("CLOUD_RUN_TASK_COUNT"),
                "missing_inputs": missing,
            },
            ensure_ascii=False,
        )
    )

    if missing:
        raise SystemExit(
            "Missing required inputs: "
            + ", ".join(missing)
            + "\nProvide CLI flags or the corresponding env vars."
        )

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    project_id = cfg["project_id"]
    canonical_ds = cfg["datasets"]["canonical"]
    ops_ds = cfg["datasets"]["ops"]
    tables = cfg["tables"]
    llm_cfg = cfg["llm"]

    cip_levels = _csv_ints(args.cip_levels)
    noc_levels = _csv_ints(args.noc_levels)
    max_pairs = None if args.max_pairs == 0 else args.max_pairs

    bq = bigquery.Client(project=project_id, location=cfg["location"])

    refs = DatasetRefs(
        project_id=project_id,
        canonical_dataset=canonical_ds,
        ops_dataset=ops_ds,
    )
    tref = TableRefs(
        pairs=tables["pairs"],
        cip_canonical=tables["cip_canonical"],
        noc_canonical=tables["noc_canonical"],
        results=tables["results"],
    )

    prompt = load_prompt(args.prompt_path, args.prompt_version)

    query_sql = build_inputs_query(
        refs=refs,
        tables=tref,
        run_id=args.run_id,
        prompt_version=args.prompt_version,
        cip_levels=cip_levels,
        noc_levels=noc_levels,
        n_shards=args.n_shards,
        shard_index=args.shard_index,
        max_pairs=max_pairs,
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("run_id", "STRING", args.run_id),
            bigquery.ScalarQueryParameter("prompt_version", "STRING", args.prompt_version),
        ]
    )

    rows = list(bq.query(query_sql, job_config=job_config).result())
    if not rows:
        print(f"No work found for shard {args.shard_index}/{args.n_shards}.")
        return

    llm = LLMClient(
        LLMClientConfig(
            base_url=llm_cfg["base_url"],
            generate_path=llm_cfg.get("generate_path", "/api/generate"),
            model=llm_cfg.get("model", "llama3"),
            timeout_s=int(llm_cfg.get("timeout_s", 300)),
            max_retries=int(llm_cfg.get("max_retries", 4)),

        )
    )

    writer = ResultWriter(
        bq=bq,
        target=ResultsTarget(project_id=project_id, dataset=ops_ds, table=tables["results"]),
    )

    sem = asyncio.Semaphore(int(llm_cfg.get("concurrency", 16)))

    async with aiohttp.ClientSession() as session:

        async def process_row(r) -> Dict[str, Any]:
            fields = {
                "cip_code": r["cip_code"],
                "cip_level": r["cip_level"],
                "cip_title": r["cip_title"],
                "cip_definition": r["cip_definition"],
                "cip_exclusions": r["cip_exclusions"],
                "cip_examples": r["cip_examples"],
                "noc_code": r["noc_code"],
                "noc_level": r["noc_level"],
                "noc_title": r["noc_title"],
                "noc_definition": r["noc_definition"],
                "noc_main_duties": r["noc_main_duties"],
                "noc_employment_requirements": r["noc_employment_requirements"],
                "noc_exclusions": r["noc_exclusions"],
            }

            prompt_text = render_prompt(prompt, fields)

            async with sem:
                t0 = time.time()
                resp = await llm.infer_json(session, prompt_text)
                latency_ms = int((time.time() - t0) * 1000)

            label = _safe_get(resp, "label", None)
            confidence = _safe_get(resp, "confidence", None)
            rationale = _safe_get(resp, "rationale", None)
            model_version = _safe_get(resp, "model_version", None)

            return {
                "run_id": r["run_id"],
                "pair_id": int(r["pair_id"]),
                "cip_code": r["cip_code"],
                "cip_level": int(r["cip_level"]),
                "noc_code": r["noc_code"],
                "noc_level": int(r["noc_level"]),
                "prompt_version": r["prompt_version"],
                "input_payload_hash": r["input_payload_hash"],
                "cip_title": r["cip_title"],
                "noc_title": r["noc_title"],
                "label": label,
                "confidence": float(confidence) if confidence is not None else None,
                "rationale": rationale,
                "model_version": model_version,
                "response_json": json.dumps(resp, ensure_ascii=False),
                "semantic_cosine_similarity": None,
                "created_at": None,
                "latency_ms": latency_ms,
            }

        results = await asyncio.gather(*(process_row(r) for r in rows), return_exceptions=True)

        clean: List[Dict[str, Any]] = []
        for r, out in zip(rows, results):
            if isinstance(out, Exception):
                # record a failure row instead of crashing the shard
                clean.append({
                    "run_id": r["run_id"],
                    "pair_id": int(r["pair_id"]),
                    "cip_code": r["cip_code"],
                    "cip_level": int(r["cip_level"]),
                    "noc_code": r["noc_code"],
                    "noc_level": int(r["noc_level"]),
                    "prompt_version": r["prompt_version"],
                    "input_payload_hash": r["input_payload_hash"],
                    "cip_title": r["cip_title"],
                    "noc_title": r["noc_title"],
                    "label": None,
                    "confidence": None,
                    "rationale": None,
                    "model_version": None,
                    "response_json": json.dumps({"error": str(out)}, ensure_ascii=False),
                    "semantic_cosine_similarity": None,
                    "created_at": None,
                    "latency_ms": None,
                })
            else:
                clean.append(out)

        results = clean

    batch_size = int(llm_cfg.get("batch_size", 50))
    for i in range(0, len(results), batch_size):
        writer.insert_rows(results[i : i + batch_size])

    print(f"Inserted {len(results)} results for shard {args.shard_index}/{args.n_shards}.")


if __name__ == "__main__":
    asyncio.run(main())