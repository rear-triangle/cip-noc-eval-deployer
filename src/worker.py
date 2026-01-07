# src/worker.py

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
import yaml
from google.cloud import bigquery

from src.bq_queries import DatasetRefs, TableRefs, build_inputs_query
from src.llm_client import LLMClient, LLMClientConfig
from src.prompt_loader import load_prompt, render_prompt
from src.result_writer import ResultWriter, ResultsTarget


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--prompt-path", required=True)
    p.add_argument("--prompt-version", required=True)
    p.add_argument("--cip-levels", required=True, help="comma-separated, e.g. 3 or 1,2,3")
    p.add_argument("--noc-levels", required=True, help="comma-separated, e.g. 5 or 4,5")
    p.add_argument("--n-shards", type=int, required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--max-pairs", type=int, default=0, help="0 means no cap")
    return p.parse_args()


def _csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _safe_get(obj: Dict[str, Any], key: str, default=None):
    return obj.get(key, default)


def _now_rfc3339() -> str:
    # BigQuery accepts RFC3339 strings for TIMESTAMP fields when using insert_rows_json
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _get_llm_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compatibility shim:
    - Your current configs/dev.yaml has llm.endpoint_url
    - The Ollama-correct setup uses llm.base_url + llm.generate_path + llm.model

    We'll support both so you can iterate without re-editing everything at once.
    """
    llm_cfg = dict(cfg.get("llm", {}))

    # If user hasn't updated config yet, fall back to endpoint_url as base_url.
    if "base_url" not in llm_cfg and "endpoint_url" in llm_cfg:
        llm_cfg["base_url"] = llm_cfg["endpoint_url"]

    # Defaults for Ollama
    llm_cfg.setdefault("generate_path", "/api/generate")

    # Model must be set eventually; defaulting is risky, but better than crashing immediately.
    llm_cfg.setdefault("model", "llama3")

    # Reasonable defaults
    llm_cfg.setdefault("timeout_s", 300)
    llm_cfg.setdefault("max_retries", 4)
    llm_cfg.setdefault("concurrency", 2)
    llm_cfg.setdefault("insert_batch_size", 50)

    return llm_cfg


async def main() -> None:
    args = _parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    project_id = cfg["project_id"]
    location = cfg["location"]
    canonical_ds = cfg["datasets"]["canonical"]
    ops_ds = cfg["datasets"]["ops"]
    pairs_ds = cfg["datasets"].get("pairs", ops_ds)

    tables = cfg["tables"]
    llm_cfg = _get_llm_cfg(cfg)

    cip_levels = _csv_ints(args.cip_levels)
    noc_levels = _csv_ints(args.noc_levels)
    max_pairs: Optional[int] = None if args.max_pairs == 0 else args.max_pairs

    bq = bigquery.Client(project=project_id, location=location)

    refs = DatasetRefs(
        project_id=project_id,
        canonical_dataset=canonical_ds,
        ops_dataset=ops_ds,
        pairs_dataset=pairs_ds,
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
        print("No work found for this shard (already complete, empty slice, or capped).")
        return

    llm = LLMClient(LLMClientConfig(
        base_url=llm_cfg["base_url"],
        generate_path=llm_cfg.get("generate_path", "/api/generate"),
        model=llm_cfg.get("model", "llama3"),
        timeout_s=int(llm_cfg.get("timeout_s", 300)),
        max_retries=int(llm_cfg.get("max_retries", 4)),
        impersonate_service_account=llm_cfg.get("impersonate_service_account"),
    ))

    writer = ResultWriter(
        bq=bq,
        target=ResultsTarget(project_id=project_id, dataset=ops_ds, table=tables["results"]),
    )

    sem = asyncio.Semaphore(int(llm_cfg.get("concurrency", 2)))

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

            # Extract structured outputs (fallback-safe)
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
                "semantic_cosine_similarity": None,  # fill later when embeddings exist
                "created_at": _now_rfc3339(),
                "latency_ms": latency_ms,
            }

        # Process rows concurrently
        results: List[Dict[str, Any]] = await asyncio.gather(*(process_row(r) for r in rows))

    # Insert in batches
    batch_size = int(llm_cfg.get("insert_batch_size", 50))
    for i in range(0, len(results), batch_size):
        writer.insert_rows(results[i:i + batch_size])

    print(f"Inserted {len(results)} results for shard {args.shard_index}/{args.n_shards}.")


if __name__ == "__main__":
    asyncio.run(main())
