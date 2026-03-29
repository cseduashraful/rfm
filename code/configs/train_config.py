TRAIN_CONFIG = {
    "name": "default_train_config",
    "batch_size": 16,
    "history_length": 10,
    "history_sampling_strategy": "random_prior",
    "history_source": "dataset", #"task_table",
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
