# src/result_writer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from google.cloud import bigquery


@dataclass(frozen=True)
class ResultsTarget:
    project_id: str
    dataset: str
    table: str


class ResultWriter:
    def __init__(self, bq: bigquery.Client, target: ResultsTarget):
        self.bq = bq
        self.target = target

    def insert_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        table_id = f"{self.target.project_id}.{self.target.dataset}.{self.target.table}"

        # Deterministic insertIds for de-duplication on retries
        # BigQuery streaming insert can de-dupe best-effort when insertId matches.
        row_ids = [f"{r['run_id']}:{r['pair_id']}" for r in rows]

        errors = self.bq.insert_rows_json(table_id, rows, row_ids=row_ids)
        if errors:
            raise RuntimeError(f"BigQuery insert errors: {errors[:3]}")