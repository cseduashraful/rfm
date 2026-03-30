# RFM Code

This folder contains the data loading, preprocessing, and training-entry code
for the RFM prototype built on top of RelBench.

## Environment

Use the `rfm` environment:

```bash
cd /work/pi_mserafini_umass_edu/ashraful/RFM
source ~/envs/rfm/bin/activate
```

All commands below assume you run them from the repo root:

```bash
cd /work/pi_mserafini_umass_edu/ashraful/RFM
```

## Main Files

- `code/configs/train_config.py`
  Main experiment config. Edit this first.
- `code/preprocess_train.py`
  Precompute and save train data plus history candidate caches.
- `code/train.py`
  Load preprocessed artifacts and iterate through epochs/batches.
- `code/train_data.py`
  Core train-data utilities, history construction, and cache loading.
- `code/data_cli.py`
  Small CLI for inspecting RelBench datasets through DuckDB.

## Recommended Workflow

### 1. Edit the Train Config

The default config lives at:

```bash
code/configs/train_config.py
```

Important settings:

- `datasets`
  Which RelBench datasets/tasks to include.
- `batch_size`
  Number of examples per batch.
- `history_length`
  Number of prior history items to sample per example.
- `history_source`
  `task_table` or `dataset`.
- `history_parallel_mode`
  `grouped_vectorized` or `multiprocess`.
- `cache_train_history_candidates`
  Whether to save and reuse compact train-history candidate pools.

### 2. Preprocess Once

This step builds and saves:

- train tables for each `(dataset, task)`
- cached history candidate pools
- a manifest describing the preprocessed output

Run:

```bash
python code/preprocess_train.py --config code/configs/train_config.py
```

If you want to force cache-only behavior for RelBench:

```bash
python code/preprocess_train.py \
  --config code/configs/train_config.py \
  --no-dataset-download \
  --no-task-download
```

Preprocessed output is saved under:

```bash
artifacts/preprocessed_train/<config_name>/
```

### 3. Run Training Loop / Batch Iteration

After preprocessing, run:

```bash
python code/train.py --config code/configs/train_config.py
```

This script:

- loads the preprocessed artifacts
- constructs a train DataLoader
- builds history for each batch from the saved candidate pools
- iterates for the configured number of epochs
- prints epoch timing

Useful options:

```bash
python code/train.py --config code/configs/train_config.py --epochs 10
python code/train.py --config code/configs/train_config.py --num-workers 4
python code/train.py --config code/configs/train_config.py --preview-batches 1
```

## Dataset Inspection Utilities

You can inspect a RelBench dataset through DuckDB with:

```bash
python code/data_cli.py --dataset rel-amazon tables
python code/data_cli.py --dataset rel-amazon sql --query "select * from relbench_meta.table_info"
```

This is useful for understanding raw database structure before changing history
construction or adding new datasets.

## Current Train Data Semantics

Each train batch item contains:

- `dataset`
- `task`
- `split`
- `row_index`
- `timestamp`
- `output`
- `train_item`

History is constructed afterward from the batch by calling logic in
`code/train_data.py`.

Depending on config:

- `history_source="task_table"`
  History is sampled from earlier rows already present in the task train table.

- `history_source="dataset"`
  History is built from task-valid prior timestamps derived from the dataset/task
  definition, using cached candidate pools when available.

## Parallel History Modes

Configured in `train_config.py`:

- `history_parallel_mode="grouped_vectorized"`
  Group batch items by `(dataset, task)` and process them in one process with
  lower overhead. This is usually the first mode to try.

- `history_parallel_mode="multiprocess"`
  Group batch items by `(dataset, task)` and process groups with CPU worker
  processes. Worker count is controlled by `history_parallel_workers`.

## Cache Behavior

There are two relevant cache ideas in the current pipeline:

- `cache_dataset_history_labels`
  In-memory timestamp-level label cache used in dataset-history logic.

- `cache_train_history_candidates`
  Disk-backed cache of compact train-history candidate pools. This is the main
  reusable cache for speeding up later runs and later epochs.

If preprocessing has already been run and the config has not changed in a
cache-breaking way, training should reuse the saved candidate cache.

## Typical Commands

Preprocess:

```bash
python code/preprocess_train.py --config code/configs/train_config.py --no-dataset-download --no-task-download
```

Train:

```bash
python code/train.py --config code/configs/train_config.py --epochs 10
```

Inspect dataset metadata:

```bash
python code/data_cli.py --dataset rel-amazon sql --query "select * from relbench_meta.table_info"
```

## Notes

- RelBench dataset/task caches usually live under `~/.cache/relbench/`.
- DuckDB artifacts are written under `artifacts/duckdb/`.
- Preprocessed training artifacts are written under `artifacts/preprocessed_train/`.
- These generated outputs are ignored by the repo `.gitignore`.
