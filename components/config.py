cfg = {
    # general setting
    "batch_size": 32,
    "input_size": (240, 320),  # (h,w)

    # training dataset
    "dataset_path": 'dataset/train_mask.tfrecord',  # 'dataset/trainval_mask.tfrecord'
    "val_path": 'dataset/val_mask.tfrecord',  #
    "dataset_len": 6115,  # train 6115 , trainval 7954, number of training samples
    "val_len": 1839,
    "using_crop": True,
    "using_bin": True,
    "using_flip": True,
    "using_distort": True,
    "using_normalizing": True,
    "labels_list": ['background', 'mask', 'unmask'],  # xml annotation

    # anchor setting
    # "min_sizes": [[(9, 7), (24, 20), (39, 35)], [(54, 41), (65, 61), (81, 66)],
    #               [(94, 86), (113, 95), (131, 122)], [(137, 128), (172, 162), (176, 210)]],
    "min_sizes":[[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    "steps": [8, 16, 32, 64],
    "match_thresh": 0.45,
    "variances": [0.1, 0.2],
    "clip": False,

    # network
    "base_channel": 16,

    # training setting
    "resume": False,  # if False,training from scratch
    "epoch": 100,
    "init_lr": 1e-2,
    "lr_decay_epoch": [50, 70],
    "lr_rate": 0.1,
    "warmup_epoch": 5,
    "min_lr": 1e-4,

    "weights_decay": 5e-4,
    "momentum": 0.9,
    "save_freq": 1, #frequency of save model weights

    # inference
    "score_threshold": 0.5,
    "nms_threshold": 0.4,
    "max_number_keep": 200
}
