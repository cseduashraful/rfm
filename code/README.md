# RFM Code

This folder contains the first-stage data pipeline for RelBench datasets.

Use the `rfm` environment and run the script directly:

```bash
~/envs/rfm/bin/python /work/pi_mserafini_umass_edu/ashraful/RFM/code/data_cli.py \
  --dataset rel-event tables
```

By default, the loader uses RelBench's own cache location and asks RelBench to
verify/download the prepared cache if needed. If a dataset is already cached,
RelBench should reuse it rather than redownload it.

Common commands:

```bash
~/envs/rfm/bin/python /work/pi_mserafini_umass_edu/ashraful/RFM/code/data_cli.py \
  --dataset rel-amazon materialize

~/envs/rfm/bin/python /work/pi_mserafini_umass_edu/ashraful/RFM/code/data_cli.py \
  --dataset rel-amazon tables

~/envs/rfm/bin/python /work/pi_mserafini_umass_edu/ashraful/RFM/code/data_cli.py \
  --dataset rel-amazon sql --query "select * from relbench_meta.table_info"
```

Config-driven dataset/task loading:

```bash
cd /work/pi_mserafini_umass_edu/ashraful/RFM
python code/load_train_config.py

python code/load_train_config.py --config code/configs/train_config.py

python code/load_train_config.py --config code/configs/train_config.py --no-dataset-download --no-task-download
```

The default config file is:

```bash
code/configs/train_config.py
```

It currently loads:
- `rel-amazon` with task `user-ltv`
- `rel-f1` with task `driver-position`

Training loader preview:

```bash
cd /work/pi_mserafini_umass_edu/ashraful/RFM
python code/train.py --config code/configs/train_config.py
```

The train config now also defines `batch_size`. The mixed loader in
`code/train.py` creates shuffled batches across all configured dataset/task
pairs. Each instance in a batch contains:
- `dataset`
- `task`
- `split`
- `timestamp`
- `output`
- `train_item`
- `history`

`history` is a task-specific list of up to `history_length` prior examples for
the same entity `x`, currently sampled by randomly choosing prior timestamps
from that task's train table. Each history item contains:
- `x`
- `timestamp`
- `output`
- `train_item`

Most of the train data loading logic now lives in:

```bash
code/train_data.py
```

`code/train.py` is kept as a small entrypoint for loading the config, building
the loader, and previewing batches.

Behavior:

- RelBench materializes/caches dataset parquet tables under its configured cache directory, usually `~/.cache/relbench/<dataset>/db`.
- DuckDB creates a database file under `artifacts/duckdb/<dataset>.duckdb`.
- Each table is exposed both as `<table_name>` and `relbench.<table_name>`.
