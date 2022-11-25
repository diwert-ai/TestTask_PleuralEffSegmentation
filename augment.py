from config import Config
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def make_train_augmenter():
    p = Config.aug_prob
    crop_size = round(Config.image_size*Config.crop_size)

    if p <= 0:
        return A.Compose([
            A.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
            ToTensorV2(transpose_mask=True)
        ])

    aug_list = []
    if Config.max_cutout > 0:
        aug_list.extend([
            A.CoarseDropout(
                max_holes=Config.max_cutout, min_holes=1,
                max_height=crop_size//10, max_width=crop_size//10,
                min_height=4, min_width=4, mask_fill_value=0, p=0.2*p),
        ])

    aug_list.extend([
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=25,
            interpolation=cv2.INTER_AREA, p=p),
        A.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
        A.HorizontalFlip(p=0.5*p),
        A.OneOf([
            A.MotionBlur(p=0.2*p),
            A.MedianBlur(blur_limit=3, p=0.1*p),
            A.Blur(blur_limit=3, p=0.1*p),
        ], p=0.2*p),
        A.Perspective(p=0.2*p),
    ])

    if Config.strong_aug:
        aug_list.extend([
            A.GaussNoise(var_limit=0.001, p=0.2*p),
            A.OneOf([
                A.OpticalDistortion(p=0.3*p),
                A.GridDistortion(p=0.1*p),
                A.PiecewiseAffine(p=0.3*p),
            ], p=0.2*p),
            A.OneOf([
                A.Sharpen(p=0.2*p),
                A.Emboss(p=0.2*p),
                A.RandomBrightnessContrast(p=0.2*p),
            ], p=0.3*p),
        ])

    aug_list.extend([
        ToTensorV2(transpose_mask=True)
    ])

    return A.Compose(aug_list)


def make_test_augmenter():
    crop_size = round(Config.image_size*Config.crop_size)
    return A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size),
        ToTensorV2(transpose_mask=True)
    ])
