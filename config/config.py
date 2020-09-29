from omegaconf import OmegaConf
conf_dict = {
    "experiment_name": "pet_cls_1",
    "img_dir": "./dataset/images",
    "model_dir": "outputs/train",
    "model_weight_prefix": "ped_cls_",
    "MLFLOW_TRACKING_URI": "http://localhost:5000",
    "train": {
        "trial_size": 50,
        "val_rate": 0.2,
        "num_epochs": 5,
        "batch_size": 20,
        "learning_rate": 0.001,
        "sd_patience": 5,  # スケジューラ
        "early_stopping": 10,
        # 高速化のため
        "num_worker": 2,
        "pin_memory": True,  # CPUのメモリ領域がページングされないようになり、高速化が期待できる
    },
    "model": {
        "input_size": 64,
        "kernel_size": 3,
        "conv_num": 3,
        "mid_units": [100, 100],
        "num_filters": [16, 128, 128],
        "label_num": 37,
    }
}

cnf = OmegaConf.create(conf_dict)
