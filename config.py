import torch
import numpy as np


class Config:
    debug = True
    batch_size = 2 if debug else 32
    image_size = 512
    num_workers = 0 if debug else 2
    seed = 1276312
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_arch = 'Unet'
    model_backbone = 'efficientnet-b0'
    num_classes = 1
    in_channels = 5
    pretrained = True
    epochs = 3 if debug else 30
    val_freq = 1
    scheduler = 'ExponentialLR'
    lr = 0.001
    min_lr = 1e-6
    wd = 1e-6
    t_max = int(30000/batch_size*epochs)+50
    t_0 = 25
    warmup_epochs = 0
    aug_prob = 0.3
    crop_size = 1.0
    max_cutout = 20
    strong_aug = False
    worker_init_fn = None if debug else (lambda x: np.random.seed(torch.initial_seed() // 2 ** 32 + x))
