# Phase 3 Pipeline

Phase 3 runs DFS-table-only zero-shot inference on the **test split**.

## Run

From repo root (with `~/envs/llm` activated):

```bash
python RFM/v2/phase3/phase3_pipeline.py \
  --dataset rel-f1 \
  --task driver-position \
  --phase2-artifacts-dir RFM/v2/phase2/artifacts \
  --output-dir RFM/v2/phase3/artifacts \
  --model-size 8b \
  --llm-batch-size 2 \
  --history-length 10
```

Example with test subset:

```bash
python RFM/v2/phase3/phase3_pipeline.py \
  --dataset rel-trial \
  --task study-adverse \
  --phase2-artifacts-dir RFM/v2/phase2/artifacts \
  --output-dir RFM/v2/phase3/artifacts \
  --model-size 8b \
  --max-items 100
```

## Design

- Uses test split rows as query items.
- For each query timestamp, only same-entity history rows with `timestamp < query_timestamp` are included.
- Prompt mode is DFS-table only (`include_neighbors=False`, `include_dfs_summary=False`, `include_dfs_table=True`).
- FastDFS feature dicts are filtered using Phase 2 outputs:
  - `task_spec.json` (`attribute_aggregation_plan`)
  - `attribute_importance.json` (if available)

## Outputs

Per dataset/task under `<output-dir>/<dataset>/<task>/`:

- `phase3_summary.json`
- `phase3_predictions.json`

