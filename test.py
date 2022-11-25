import pandas as pd
import torch.utils.data as data
from dataset import PleuralEffDataset
from config import Config
import matplotlib.pyplot as plt
from augment import make_train_augmenter


def test_pl_dataset():
    df_train = pd.read_csv('e:/train/train.csv')
    dataset = PleuralEffDataset(df_train, transform=make_train_augmenter())
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


if __name__ == '__main__':
    test_pl_dataset()
