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
- `code/configs/inference_config.py`
  Main zero-shot inference config for single-dataset, single-task runs.
- `code/preprocess_train.py`
  Precompute and save train data plus history candidate caches.
- `code/train.py`
  Load preprocessed artifacts and iterate through epochs/batches.
- `code/train_data.py`
  Core train-data utilities, history construction, and cache loading.
- `code/zero_shot.py`
  Zero-shot runtime entrypoint. Builds per-query histories and prompts, runs the
  local LLM, and reports MAE when labels are available.
- `code/zero_shot_llm.py`
  Local LLaMA wrapper, model-path registry, batched generation, and numeric
  extraction helper.
- `code/inference_history.py`
  Inference-time history construction from raw RelBench data or task tables.
- `code/grag.py`
  GraphRAG store, temporal history index, k-hop neighbor retrieval, prompt
  construction, and optional semantic similarity retrieval.
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

## Zero-Shot Inference

The repository also contains a zero-shot inference path that does not depend on
the preprocessed train artifacts. Instead, it constructs history and graph
context online for each evaluation query and feeds a prompt to a locally
downloaded LLaMA model.

### Entry Point

Run:

```bash
python code/zero_shot.py
```

This script:

- loads `code/configs/inference_config.py`
- materializes the requested RelBench dataset and task
- constructs per-query task-valid history online
- builds GraphRAG context from temporal PK/FK structure
- formats a prompt for each query
- runs local LLaMA inference
- computes MAE if labels are available

### Files Involved

- `code/zero_shot.py`
  Main entrypoint, CLI, runtime loop, batching, streaming, logging, and MAE.
- `code/zero_shot_llm.py`
  LLM wrapper. Supports local `1b`, `3b`, and `8b` LLaMA checkpoints.
- `code/inference_history.py`
  Constructs zero-shot history online for one query at a time.
- `code/grag.py`
  Builds relational graph indexes, retrieves temporal neighbors, and formats the
  final prompt.

### Zero-Shot Config

The default inference config lives at:

```bash
code/configs/inference_config.py
```

Important fields:

- `dataset`
  Single RelBench dataset name.
- `task`
  Single RelBench task name.
- `history_length`
  Maximum number of task-history examples to include.
- `history_sampling_strategy`
  One of:
  - `recent_min_overlap`
  - `most_recent_k`
  - `random_prior`
- `history_source`
  `dataset` or `task_table`
- `use_full_raw_db`
  Whether inference should use the full raw RelBench DB rather than the
  benchmark-truncated DB.
- `include_future_rows`
  Must remain compatible with `use_full_raw_db`; this enables access to raw rows
  later than the benchmark split boundary while still filtering by query time at
  retrieval time.

### Data Semantics During Zero-Shot Inference

Each inference query is a task row:

- one entity id
- one timestamp
- optionally one target label if evaluating on `val` or unmasked `test`

For a query `(x, t)`, the pipeline first builds a history of prior task-shaped
rows for the same entity.

For `history_source="dataset"`:

- candidate timestamps are drawn from raw DB tables with time columns
- only timestamps strictly before the query time are considered
- `task.make_table(db, candidate_timestamps)` is used to turn raw timestamps
  into task-valid rows
- rows are filtered to the same entity
- rows are filtered again to satisfy the non-overlap constraint with the query:
  `history_time + task.timedelta <= query_time`
- duplicates on `(entity, timestamp)` are removed
- a history sampler chooses up to `history_length` rows

For `history_source="task_table"`:

- history is sampled directly from earlier task rows already present in the task
  table

### History Sampling Strategies

Zero-shot inference currently supports three samplers:

- `most_recent_k`
  Choose the most recent `k` valid history rows.
- `random_prior`
  Uniformly sample `k` rows from all valid prior history rows.
- `recent_min_overlap`
  Prefer recent rows, but greedily try to keep selected timestamps at least one
  label horizon apart before filling remaining slots.

The current default is deterministic and label-safe:

```bash
history_sampling_strategy = "most_recent_k"
```

unless changed in the config or overridden on the CLI.

### GraphRAG Retrieval During Zero-Shot Inference

The zero-shot path builds a GraphRAG store over the raw RelBench database.

The store contains:

- node ids for table primary-key rows
- edge types for FK-to-PK relations
- forward adjacency
- reverse adjacency
- a temporal history index for rows involving the task entity

This means retrieval is not limited to one FK direction. The graph traversal can
follow:

- outgoing FK-to-PK edges
- reverse incoming links

so query-time neighborhoods are effectively bidirectional over the relational
schema.

### Query Context Construction

For each prompt entrypoint, `grag.py` currently constructs:

- static rows for the entity
- temporal history rows for the entity before the entrypoint cutoff
- k-hop neighbors from those history rows
- k-hop neighbors from the query entity node itself
- optionally same-table semantic-similarity retrieval when enabled

Important current rule:

- query-node k-hop neighbors are always considered
- history-derived neighbors are also included when history exists
- all temporal neighbor retrieval is cutoff-aware

### Prompt Structure

The prompt is designed for a decoder-only local LLM and currently contains:

- task header
- explicit instruction to return only a number
- one short explanation of what context is included
- one static block
- zero or more prior history examples
- per-example history and neighbor context
- optional semantic similarity block if enabled
- one final query block
- a short direct prediction instruction
- `Answer:`

History examples are labeled with their known outputs. The final query omits the
output and asks the LLM to predict it.

### Neighbor Rendering

All retrieved neighbors are currently included in the prompt. The prompt builder
does not perform prompt-side neighbor truncation beyond whatever was already
retrieved by the GraphRAG search itself.

Neighbor retrieval is controlled by:

- `--num-hops`
- `--top-k`

where:

- `num_hops` caps traversal depth
- `top_k` caps the number of neighbors retrieved per expanded node

### Semantic Retrieval Flag

The zero-shot runtime also exposes an optional generic semantic retrieval path:

```bash
python code/zero_shot.py --semantic-retrieval
```

When enabled, GraphRAG additionally retrieves same-table similar entities using
generic static-attribute similarity and includes:

- similar entity identity
- similarity score
- static summary
- pre-cutoff history rows for the similar entity

This path is database-agnostic in design and does not require task labels, but
it is off by default.

### FastDFS / DFS Flag

The zero-shot CLI supports FastDFS-based relational feature summaries:

```bash
python code/zero_shot.py --use-dfs
```

Backward-compatible alias:

```bash
python code/zero_shot.py --use-fastdfs
```

When enabled:

- FastDFS features are computed per prompt entry timestamp
  (each history example timestamp plus the final query timestamp).
- Cutoff time is enforced in strict mode (`timestamp - 1ns`) to avoid temporal
  leakage from same-timestamp rows.
- Raw DFS columns are not injected directly; instead, prompts include
  human-readable summaries grouped by meta-path.

### Batching and Streaming

The current zero-shot runtime uses a two-stage pipeline:

1. context generation on CPU
2. LLM generation on GPU

Relevant CLI flags:

- `--context-batch-size`
  Number of examples to prepare before continuing through the pipeline.
- `--llm-batch-size`
  Number of prompts sent to the LLM at once.
- `--context-workers`
  Controls concurrency across examples and within prompt construction.
- `--frontier-workers`
  Controls concurrency while expanding frontier nodes within a graph hop.

The runtime is also streaming:

- prepared prompts are emitted to the LLM queue as they finish
- the LLM does not need to wait for the whole context batch when
  `llm_batch_size` is small

### Parallelism Notes

Current parallelism exists at multiple levels:

- example-level context preparation inside a context batch
- per-entry context construction inside one prompt
- per-hop frontier expansion inside multi-hop retrieval

This can improve throughput, but depending on the machine and Python overhead it
can also increase latency if worker counts are too high. In practice, these
flags should be tuned together.

### Logging and Debugging

Default behavior is quiet except for final MAE or end-of-run summaries.

Useful flags:

- `--print-log`
  Print detailed per-item information.
- `--pred-only`
  With `--print-log`, print only prediction and ground truth per item.
- `--debug`
  Limit evaluation to `--max-items`.

Example:

```bash
python code/zero_shot.py --debug --max-items 10 --print-log --pred-only
```

### Local LLM Selection

Supported model sizes:

- `1b`
- `3b`
- `8b`

Select them with:

```bash
python code/zero_shot.py --model-size 1b
python code/zero_shot.py --model-size 3b
python code/zero_shot.py --model-size 8b
```

The paths are configured in `code/zero_shot_llm.py`.

### Typical Zero-Shot Commands

Basic run:

```bash
python code/zero_shot.py
```

Two-hop GraphRAG with debugging:

```bash
python code/zero_shot.py --num-hops 2 --debug --print-log --pred-only
```

Two-stage batched inference:

```bash
python code/zero_shot.py \
  --num-hops 2 \
  --context-batch-size 32 \
  --llm-batch-size 4
```

Enable semantic retrieval:

```bash
python code/zero_shot.py --semantic-retrieval
```

Use a different local model:

```bash
python code/zero_shot.py --model-size 8b
```

### Current Limitations of Zero-Shot Inference

- dataset-based history construction is still expensive because it builds
  task-valid rows online
- multi-hop retrieval can become slow for large `top_k` or many history
  examples
- prompt size can grow quickly when all neighbors are included
- same-table semantic retrieval is generic but still heuristic
- throughput and latency depend strongly on worker settings

Even with these limitations, the zero-shot path is now a complete inference
stack:

- raw RelBench DB access
- history construction
- GraphRAG retrieval
- prompt construction
- local LLaMA inference
- MAE evaluation

## Notes

- RelBench dataset/task caches usually live under `~/.cache/relbench/`.
- DuckDB artifacts are written under `artifacts/duckdb/`.
- Preprocessed training artifacts are written under `artifacts/preprocessed_train/`.
- These generated outputs are ignored by the repo `.gitignore`.
