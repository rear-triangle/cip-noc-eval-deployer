from __future__ import annotations
import argparse
import uuid
import yaml
from datetime import datetime, timezone
from typing import List

from google.cloud import bigquery

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--prompt-path", required=True)
    p.add_argument("--prompt-version", required=True)
    p.add_argument("--cip-levels", required=True)
    p.add_argument("--noc-levels", required=True)
    p.add_argument("--n-shards", type=int, default=0)
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--notes", default="")
    return p.parse_args()

def _csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main() -> None:
    args = _parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    project_id = cfg["project_id"]
    location = cfg["location"]
    ops_ds = cfg["datasets"]["ops"]
    run_registry = cfg["tables"]["run_registry"]

    cip_levels = _csv_ints(args.cip_levels)
    noc_levels = _csv_ints(args.noc_levels)

    n_shards = args.n_shards if args.n_shards > 0 else int(cfg["run_defaults"].get("n_shards", 50))
    max_pairs = None if args.max_pairs == 0 else args.max_pairs

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

    bq = bigquery.Client(project=project_id, location=location)

    # Insert run metadata
    table_id = f"{project_id}.{ops_ds}.{run_registry}"
    row = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": args.prompt_version,
        #"prompt_profile": cfg["run_defaults"].get("prompt_profile", "balanced"),
        "model_endpoint": cfg["llm"]["base_url"],
        "model_version": None,
        "cip_version": "CIP_2021_v1_0",
        "noc_version": "NOC_2021_v1_0",
        "cip_levels": cip_levels,
        "noc_levels": noc_levels,
        "n_shards": n_shards,
        "max_pairs": max_pairs,
        "notes": args.notes,
    }
    errors = bq.insert_rows_json(table_id, [row])
    if errors:
        raise RuntimeError(f"Failed inserting run metadata: {errors}")

    print(f"Created run_id: {run_id}")
    print("\nRun workers like:\n")

    for shard_index in range(n_shards):
        print(
            "python -m src.worker "
            f'--config "{args.config}" '
            f'--run-id "{run_id}" '
            f'--prompt-path "{args.prompt_path}" '
            f'--prompt-version "{args.prompt_version}" '
            f'--cip-levels "{args.cip_levels}" '
            f'--noc-levels "{args.noc_levels}" '
            f'--n-shards {n_shards} '
            f'--shard-index {shard_index} '
            f'--max-pairs {args.max_pairs}'
        )

if __name__ == "__main__":
    main()
