# Phase 1 Pipeline

This folder contains Phase 1 implementation for relational DB intelligence extraction.

The current implementation is SQL-first for large-table scalability: profiling runs over
DuckDB/parquet-backed relations and only pulls bounded samples into pandas when needed.

## Run

From repo root (with `~/envs/llm` activated):

```bash
python RFM/v2/phase1/phase1_pipeline.py \
  --dataset rel-f1 \
  --output-dir RFM/v2/phase1/artifacts \
  --max-path-depth 8
```

Optional config-file driven run:

```bash
python RFM/v2/phase1/phase1_pipeline.py \
  --config RFM/v2/phase1/phase1_config_template.yaml
```

LLM-based semantics labeling is enabled by default. Use `--disable-llm-semantics` if you want a heuristic-only run.

## Outputs

For each dataset, the pipeline writes:

- `schema.json`
- `stats.json`
- `semantics.json`
- `semantic_context_graph.json`
- `path_catalog.json`
- `safety_rules.json`

under:

- `<output-dir>/<dataset>/`

## Notes

- Enforces global temporal cutoff from earliest `val_start` across tasks in the dataset.
- Uses sampled profiling for large tables by default and records sampling metadata.
- Records `exact` vs `sampled` profiling mode plus cutoff audit counts per table.
- Low-confidence inferred FK links are retained in full catalog but excluded from recommended paths.
- `path_catalog.json` is optional and currently a lightweight placeholder when enabled.

## Code Structure

- `phase1_pipeline.py`: orchestration and artifact assembly.
- `phase1_data_access.py`: lazy DuckDB/parquet-backed table access and cutoff-aware relations.
- `phase1_stats_sql.py`: SQL-backed profiling helpers with bounded sampling.
- `phase1_utils.py`: JSON/statistical helpers and semantic role constants.
- `phase1_fk.py`: FK inference helpers.
- `phase1_semantic_graph.py`: semantic context graph construction logic.

## Technical Report

- `technical_report.tex` provides a concise technical description of the Phase 1 design and extension points.
