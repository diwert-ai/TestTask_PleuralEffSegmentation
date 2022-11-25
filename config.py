import torch


class Config:
    batch_size = 32
    image_size = 512
    num_workers = 2
    seed = 1276312
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_arch = 'Unet'
    model_backbone = 'efficientnet-b0'
    num_classes = 1
    in_channels = 5
    pretrained = True
    epochs = 30
    val_freq = 1
    scheduler = 'ExponentialLR'
    debug = False
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
