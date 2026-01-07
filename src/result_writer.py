from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
        errors = self.bq.insert_rows_json(table_id, rows)
        if errors:
            # Errors is a list of dicts; raise with some detail.
            raise RuntimeError(f"BigQuery insert errors: {errors[:3]}")
