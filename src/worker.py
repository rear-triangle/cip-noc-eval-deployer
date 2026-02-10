# src/worker.py

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
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


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def _csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _safe_get(obj: Dict[str, Any], key: str, default=None):
    return obj.get(key, default)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--config", default=_env("CONFIG_PATH"))
    p.add_argument("--run-id", default=_env("RUN_ID"))

    p.add_argument("--prompt-path", default=_env("PROMPT_PATH"))
    p.add_argument("--prompt-version", default=_env("PROMPT_VERSION"))

    p.add_argument("--cip-levels", default=_env("CIP_LEVELS"), help="comma-separated, e.g. 3 or 1,2,3")
    p.add_argument("--noc-levels", default=_env("NOC_LEVELS"), help="comma-separated, e.g. 5 or 4,5")

    p.add_argument("--n-shards", type=int, default=int(os.getenv("CLOUD_RUN_TASK_COUNT", "1")))
    p.add_argument("--shard-index", type=int, default=int(os.getenv("CLOUD_RUN_TASK_INDEX", "0")))

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


async def _wait_for_llm_ready(llm: LLMClient, session: aiohttp.ClientSession, prompt_text: str, ready_retries: int, ready_sleep_s: int) -> None:
    last_err: Optional[Exception] = None
    for i in range(1, ready_retries + 1):
        try:
            _ = await llm.infer_json(session, prompt_text)
            print(json.dumps({"msg": "llm ready", "ready_attempt": i}))
            return
        except Exception as e:
            last_err = e
            print(json.dumps({"msg": "llm not ready yet", "ready_attempt": i, "ready_retries": ready_retries, "sleep_s": ready_sleep_s, "error": str(e)}))
            await asyncio.sleep(ready_sleep_s)
    raise RuntimeError(f"LLM never became ready after {ready_retries} attempts. Last error: {last_err}")


async def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    missing = _validate_args(args)

    print(json.dumps({
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
        "missing_inputs": missing,
    }, ensure_ascii=False))

    if missing:
        raise SystemExit("Missing required inputs: " + ", ".join(missing) + "\nProvide CLI flags or the corresponding env vars.")

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    cip_levels = _csv_ints(args.cip_levels)
    noc_levels = _csv_ints(args.noc_levels)
    max_pairs = None if args.max_pairs == 0 else args.max_pairs

    bq = bigquery.Client(project=cfg["project_id"], location=cfg["location"])

    refs = DatasetRefs(
        project_id=cfg["project_id"],
        canonical_dataset=cfg["datasets"]["canonical"],
        ops_dataset=cfg["datasets"]["ops"],
    )
    tref = TableRefs(
        pairs=cfg["tables"]["pairs"],
        cip_canonical=cfg["tables"]["cip_canonical"],
        noc_canonical=cfg["tables"]["noc_canonical"],
        results=cfg["tables"]["results"],
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
    print(json.dumps({"msg": "rows fetched", "shard_index": args.shard_index, "n_rows": len(rows)}))
    if not rows:
        print(f"No work found for shard {args.shard_index}/{args.n_shards}. run_id={args.run_id}")
        return

    llm_cfg = cfg["llm"]

    llm = LLMClient(
        LLMClientConfig(
            base_url=llm_cfg["base_url"],
            generate_path=llm_cfg.get("generate_path", "/api/generate"),
            model=llm_cfg.get("model", "llama3"),
            timeout_s=int(llm_cfg.get("timeout_s", 1800)),
            max_retries=int(llm_cfg.get("max_retries", 12)),
            temperature=float(llm_cfg.get("temperature", 0.1)),
            num_predict=int(llm_cfg.get("num_predict", 220)),
            stop_csv=str(llm_cfg.get("stop_csv", "```,")),
        )
    )

    writer = ResultWriter(
        bq=bq,
        target=ResultsTarget(project_id=cfg["project_id"], dataset=cfg["datasets"]["ops"], table=cfg["tables"]["results"]),
    )

    sem = asyncio.Semaphore(int(llm_cfg.get("concurrency", 1)))

    stagger_s = int(llm_cfg.get("stagger_s", 0) or 0)
    ready_retries = int(llm_cfg.get("ready_retries", 0) or 0)
    ready_sleep_s = int(llm_cfg.get("ready_sleep_s", 5) or 5)

    async with aiohttp.ClientSession() as session:
        if stagger_s > 0:
            delay = random.uniform(0, float(stagger_s))
            print(json.dumps({"msg": "staggering start (jitter)", "sleep_s": round(delay, 2)}, ensure_ascii=False))
            await asyncio.sleep(delay)

        if ready_retries > 0:
            warm_fields = {
                "cip_code": "00.0000",
                "cip_level": 0,
                "cip_title": "warmup",
                "cip_definition": "warmup",
                "cip_exclusions": "",
                "cip_examples": "",
                "noc_code": "00000",
                "noc_level": 0,
                "noc_title": "warmup",
                "noc_definition": "warmup",
                "noc_main_duties": "warmup",
                "noc_employment_requirements": "warmup",
                "noc_exclusions": "",
            }
            warm_prompt = render_prompt(prompt, warm_fields)
            await _wait_for_llm_ready(llm, session, warm_prompt, ready_retries, ready_sleep_s)

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

            # Prefer parsed fields; if parse_error but salvaged exists, use salvaged fields.
            label = _safe_get(resp, "label", None)
            confidence = _safe_get(resp, "confidence", None)
            rationale = _safe_get(resp, "rationale", None)
            model_version = _safe_get(resp, "model_version", None)

            now_iso = datetime.now(timezone.utc).isoformat()

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
                "created_at": now_iso,
                "latency_ms": latency_ms,
            }

        results = await asyncio.gather(*(process_row(r) for r in rows), return_exceptions=True)

        clean: List[Dict[str, Any]] = []
        for r, out in zip(rows, results):
            now_iso = datetime.now(timezone.utc).isoformat()
            if isinstance(out, Exception):
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
                    "created_at": now_iso,
                    "latency_ms": None,
                })
            else:
                clean.append(out)

        results = clean

    batch_size = int(llm_cfg.get("batch_size", 50))
    for i in range(0, len(results), batch_size):
        writer.insert_rows(results[i : i + batch_size])

    print(json.dumps({
        "msg": "worker finished",
        "run_id": args.run_id,
        "shard_index": args.shard_index,
        "n_shards": args.n_shards,
        "inserted_n": len(results),
    }, ensure_ascii=False))
    print(f"Inserted {len(results)} results for shard {args.shard_index}/{args.n_shards}. run_id={args.run_id}")


if __name__ == "__main__":
    asyncio.run(main())