TRAIN_CONFIG = {
    "name": "default_train_config",
    "batch_size": 256,
    "history_length": 10,
    "history_sampling_strategy": "random_prior",
    "history_source": "dataset", #"task_table",
    "history_parallel_mode": "grouped_vectorized", #"multiprocess", #
    "history_parallel_workers": 8,
    "cache_dataset_history_labels": False,
    "cache_train_history_candidates": True,
    "dataset_download": True,
    "task_download": True,
    "include_future_rows": False,
    "datasets": [
        {
            "name": "rel-amazon",
            "tasks": [
                "user-ltv",
                "user-churn",
                "item-ltv",
                "item-churn",
            ],
        },
        {
            "name": "rel-f1",
            "tasks": ["driver-position"],
        },
    ],
}


# Config reference
#
# name:
#   A human-readable config name. Used to name preprocessed output folders.
#
# batch_size:
#   Number of train examples per batch.
#
# history_length:
#   Maximum number of prior history items sampled for each train example.
#
# history_sampling_strategy:
#   How to sample from the available prior history candidates.
#   Supported options:
#   - "random_prior": uniformly sample up to history_length prior candidates.
#
# history_source:
#   Where history candidates come from.
#   Supported options:
#   - "task_table": use earlier rows already present in the task's train table.
#   - "dataset": use task-valid prior timestamps derived from the raw dataset/task
#     definition, then sample from cached valid train-history candidates.
#
# history_parallel_mode:
#   Runtime strategy for constructing history from cached candidates.
#   Supported options:
#   - "grouped_vectorized": process the batch grouped by (dataset, task) in a
#     single process with lower Python overhead.
#   - "multiprocess": process grouped batch items using a CPU worker pool.
#
# history_parallel_workers:
#   Number of worker processes used when history_parallel_mode == "multiprocess".
#
# cache_dataset_history_labels:
#   Used only with history_source == "dataset".
#   If True, cache timestamp-level task labels in RAM during preprocessing/runtime.
#   If False, do not keep that lower-level in-memory label cache.
#
# cache_train_history_candidates:
#   Used only with history_source == "dataset".
#   If True, save compact train-history candidate pools to disk and reuse them
#   across batches, epochs, and later runs.
#   If False, do not persist those candidate caches.
#
# dataset_download:
#   If True, allow RelBench to verify/download prepared dataset caches during
#   preprocessing. If False, rely on existing local dataset caches only.
#
# task_download:
#   If True, allow RelBench to verify/download prepared task caches during
#   preprocessing. If False, rely on existing local task caches only.
#
# include_future_rows:
#   If False, datasets are truncated at the test timestamp to avoid leakage.
#   If True, allow rows beyond the test timestamp when materializing datasets.
#
# datasets:
#   List of dataset/task groups to include in preprocessing and training.
#   Each entry must contain:
#   - "name": RelBench dataset name, e.g. "rel-amazon", "rel-f1"
#   - "tasks": list of task names for that dataset
