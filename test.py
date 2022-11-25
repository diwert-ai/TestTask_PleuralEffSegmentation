import pandas as pd
import torch.utils.data as data
from dataset import PleuralEffDataset
from config import Config
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from augment import make_train_augmenter, make_test_augmenter
from preprocess import DataPreprocessor
import nibabel as nib
import cv2


def test_pl_dataset():
    df_train = pd.read_csv('e:/train/train.csv')
    dataset = PleuralEffDataset(df_train, transform=make_test_augmenter())
    loader = data.DataLoader(dataset,
                             sampler=data.RandomSampler(dataset),
                             batch_size=Config.batch_size,
                             drop_last=True,
                             num_workers=Config.num_workers,
                             pin_memory=True)

    def show_img(x, y=None):
        plt.imshow(x[:, :, 0], cmap='bone')
        if y is not None:
            plt.imshow(y[:, :, 0], alpha=0.4)
        plt.axis('off')

    def plot_batch(xs, ys, size=5):
        plt.figure(figsize=(5 * 5, 5))
        for idx in range(size):
            plt.subplot(1, 5, idx + 1)
            x = xs[idx, ].permute((1, 2, 0)).numpy() * 255.0
            x = x.astype('uint8')
            y = ys[idx, ].permute((1, 2, 0)).numpy() * 255.0
            show_img(x, y)
        plt.tight_layout()
        plt.show()

    batch = next(iter(loader))
    print(f'batch \'img\' dims: {batch[0].size()}\nbatch target \'mask\' dims: {batch[1].size()}')
    plot_batch(batch[0], batch[1])


def get_3d_masked_img_ani(img_3d_num, start_slice, stop_slice):
    data_p = DataPreprocessor()
    data_p.get_3d_paths()
    test_images_path = data_p.images_3d_paths[img_3d_num]
    test_masks_path = data_p.masks_3d_paths[img_3d_num]
    test_images_3d = data_p.load_dicom(test_images_path)
    max_slice_num = test_images_3d.shape[0]
    test_masks_3d = nib.load(test_masks_path)
    test_masks_3d = test_masks_3d.get_fdata().transpose(2, 0, 1)

    fig, ax = plt.subplots()
    ims = []
    lung_label = test_images_path[data_p.pref_img_len:data_p.pref_img_len + data_p.lung_len]
    plt.legend([Rectangle((0, 0), 1, 1)], [lung_label])

    for slice_num in range(min(start_slice, max_slice_num), min(stop_slice, max_slice_num)):
        img, msk = test_images_3d[slice_num], test_masks_3d[slice_num]
        msk = cv2.rotate(msk, cv2.ROTATE_90_COUNTERCLOCKWISE)
        im1 = ax.imshow(img, cmap='bone', animated=True)
        im2 = ax.imshow(msk, alpha=0.4, animated=True)
        ims.append([im1, im2])

        animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                  repeat_delay=1000)
    plt.show()


if __name__ == '__main__':
    test_pl_dataset()
    get_3d_masked_img_ani(0, 40, 60)
