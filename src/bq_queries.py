# src/bq_queries.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class DatasetRefs:
    project_id: str
    canonical_dataset: str
    ops_dataset: str


@dataclass(frozen=True)
class TableRefs:
    pairs: str
    cip_canonical: str
    noc_canonical: str
    results: str


def _sql_array_int(values: List[int]) -> str:
    # Safe because values are ints from CLI parsing
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def build_inputs_query(
    refs: DatasetRefs,
    tables: TableRefs,
    run_id: str,
    prompt_version: str,
    cip_levels: List[int],
    noc_levels: List[int],
    n_shards: int,
    shard_index: int,
    max_pairs: Optional[int] = None,
) -> str:
    """
    Returns a StandardSQL query that yields the input rows for a given shard.

    Assumptions:
    - CIP/NOC canonical tables live in refs.canonical_dataset
    - pairs + results live in refs.ops_dataset (your current setup)
    - results table contains run_id + pair_id so we can skip already-done work

    Output columns match what worker.py expects.
    """

    cip_levels_sql = _sql_array_int(cip_levels)
    noc_levels_sql = _sql_array_int(noc_levels)

    project = refs.project_id
    canon = refs.canonical_dataset
    ops = refs.ops_dataset

    # IMPORTANT: pairs are in OPS dataset in your setup
    pairs_tbl = f"`{project}.{ops}.{tables.pairs}`"
    cip_tbl = f"`{project}.{canon}.{tables.cip_canonical}`"
    noc_tbl = f"`{project}.{canon}.{tables.noc_canonical}`"
    results_tbl = f"`{project}.{ops}.{tables.results}`"

    limit_clause = ""
    if max_pairs is not None and int(max_pairs) > 0:
        limit_clause = f"\nLIMIT {int(max_pairs)}"

    sql = f"""
WITH
  filtered_pairs AS (
    SELECT
      p.pair_id,
      p.cip_code,
      p.cip_level,
      p.noc_code,
      p.noc_level,
      CONCAT(
        CAST(p.cip_level AS STRING), ':', p.cip_code, '|',
        CAST(p.noc_level AS STRING), ':', p.noc_code
      ) AS pair_key
    FROM {pairs_tbl} p
    WHERE p.cip_level IN UNNEST({cip_levels_sql})
      AND p.noc_level IN UNNEST({noc_levels_sql})
  ),

  todo AS (
    SELECT fp.*
    FROM filtered_pairs fp
    LEFT JOIN {results_tbl} r
      ON r.run_id = @run_id
     AND r.pair_id = fp.pair_id
    WHERE r.pair_id IS NULL
    ORDER BY fp.pair_id
    {limit_clause}
  ),

  sharded AS (
    SELECT *
    FROM todo
    WHERE MOD(ABS(FARM_FINGERPRINT(pair_key)), {int(n_shards)}) = {int(shard_index)}
  ),

  joined AS (
    SELECT
      @run_id AS run_id,
      @prompt_version AS prompt_version,

      s.pair_id,
      s.cip_code,
      s.cip_level,
      c.cip_title,
      c.cip_definition,
      c.cip_exclusions,
      c.cip_examples,

      s.noc_code,
      s.noc_level,
      n.noc_title,
      n.noc_definition,
      n.noc_main_duties,
      n.noc_employment_requirements,
      n.noc_exclusions,

      TO_HEX(SHA256(CONCAT(
        'prompt_version=', @prompt_version, '\\n',
        'cip_code=', s.cip_code, '\\n',
        'cip_level=', CAST(s.cip_level AS STRING), '\\n',
        'cip_title=', COALESCE(c.cip_title, ''), '\\n',
        'cip_definition=', COALESCE(c.cip_definition, ''), '\\n',
        'cip_exclusions=', COALESCE(c.cip_exclusions, ''), '\\n',
        'cip_examples=', COALESCE(c.cip_examples, ''), '\\n',
        'noc_code=', s.noc_code, '\\n',
        'noc_level=', CAST(s.noc_level AS STRING), '\\n',
        'noc_title=', COALESCE(n.noc_title, ''), '\\n',
        'noc_definition=', COALESCE(n.noc_definition, ''), '\\n',
        'noc_main_duties=', COALESCE(n.noc_main_duties, ''), '\\n',
        'noc_employment_requirements=', COALESCE(n.noc_employment_requirements, ''), '\\n',
        'noc_exclusions=', COALESCE(n.noc_exclusions, '')
      ))) AS input_payload_hash

    FROM sharded s
    JOIN {cip_tbl} c
      ON c.cip_code = s.cip_code
     AND c.cip_level = s.cip_level
    JOIN {noc_tbl} n
      ON n.noc_code = s.noc_code
     AND n.noc_level = s.noc_level
  )

SELECT * FROM joined
"""
    return sql.strip()