INFERENCE_CONFIG = {
    "name": "default_inference_config",
    "batch_size": 1,
    "history_length": 10,
    "history_sampling_strategy": "recent_min_overlap",
    "history_source": "dataset", #"task_table",
    "history_parallel_mode": "grouped_vectorized", #"multiprocess", #
    "history_parallel_workers": 8,
    "cache_dataset_history_labels": False,
    "cache_train_history_candidates": True,
    "dataset_download": True,
    "task_download": True,
    "use_full_raw_db": True,
    "include_future_rows": True,
    "dataset": "rel-f1", 
    "task": "driver-position",
}
