import os
import cv2
import numpy as np
import torch.utils.data as data
from config import Config


class PleuralEffDataset(data.Dataset):
    def __init__(self, df, transform=None, subset=100, s_deep=2):
        self.transform = transform
        self.s_deep = s_deep
        if subset != 100:
            assert subset < 100
            num_rows = df.shape[0] * subset // 100
            df = df.iloc[:num_rows]
        self.length = len(df)
        self.img_paths = df['image_path'].to_list()
        self.slice_nums = df['slice'].to_list()
        self.msk_paths = df['mask_path'].to_list()

    @staticmethod
    def load_slice(img_file, slice_num, diff):
        filename = img_file[:-7] + str(slice_num + diff).zfill(3) + '.png'
        if os.path.exists(filename):
            return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        return None

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        slice_num = self.slice_nums[index]
        # read s_deep slices into one image
        images = [self.load_slice(img_path, slice_num, i) for i in range(-self.s_deep, self.s_deep + 1)]

        for i in range(self.s_deep + 1, 2 * self.s_deep + 1):
            if images[i] is None:
                images[i] = images[i - 1]
        for i in range(self.s_deep - 1, -1, -1):
            if images[i] is None:
                images[i] = images[i + 1]

        img = np.stack(images, axis=2)
        img = img.astype(np.float32)
        max_val = img.max()
        if max_val != 0:
            img /= max_val

        msk_path = self.msk_paths[index]
        mask = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.float32)
        mask = mask.reshape(Config.image_size, Config.image_size, 1)
        result = self.transform(image=img, mask=mask)
        img, mask = result['image'], result['mask']

        return img, mask

    def __len__(self):
        return self.length
