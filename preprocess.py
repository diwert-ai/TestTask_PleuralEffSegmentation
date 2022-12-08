import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import nibabel as nib
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from config import Config


class DataPreprocessor:
    def __init__(self, paths):
        self.paths = paths
        self.images_3d_paths = []
        self.masks_3d_paths = []
        self.images_list = []
        self.masks_list = []
        self.lung_len = len('LUNG1-005')
        self.pref_img_len = len(self.paths['3d images dir'])
        self.pref_msk_len = len(self.paths['3d masks dir'])
        self.pref_len = len(self.paths['images train'])
        self.train_dir = self.paths['images train']
        self.train_dir_masks = self.paths['masks train']

    @staticmethod
    def load_dicom(directory):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        reader.SetFileNames(dicom_names)
        image_itk = reader.Execute()
        image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
        return image_zyx

    def get_3d_paths(self):
        for path in glob.glob(self.paths['3d images files']):
            if path[-4:] != 'json':
                self.images_3d_paths.append(path)

        for path in glob.glob(self.paths['3d masks files']):
            self.masks_3d_paths.append(path)

    def save_png_slices(self, save_images=True, save_masks=True):
        for path in tqdm(self.images_3d_paths, desc='slicing 3d images ', total=len(self.images_3d_paths)):
            lung_name = path[self.pref_img_len:self.pref_img_len + self.lung_len]
            images = self.load_dicom(path)
            if save_images:
                for slice_number in range(images.shape[0]):
                    file_name = f'{self.train_dir}{lung_name.lower()}-{slice_number:03}.png'
                    cv2.imwrite(file_name, images[slice_number])
                    self.images_list.append(file_name)

        for path in tqdm(self.masks_3d_paths, desc='slicing 3d masks ', total=len(self.masks_3d_paths)):
            masks = nib.load(path)
            masks = masks.get_fdata().transpose(2, 0, 1)
            lung_name = path[self.pref_msk_len:self.pref_msk_len + self.lung_len]
            if save_masks:
                for slice_number in range(masks.shape[0]):
                    mask_file_name = f'{self.train_dir_masks}{lung_name.lower()}-{slice_number:03}.png'
                    mask = cv2.rotate(masks[slice_number], cv2.ROTATE_90_COUNTERCLOCKWISE)
                    cv2.imwrite(mask_file_name, mask)
                    self.masks_list.append(mask_file_name)

        print(f'{len(self.images_list)}: total images saved')
        print(f'{len(self.masks_list)}: total masks saved')

    def make_metadata_frame(self):
        def split(gr):
            if gr == 'lung1-026':
                return 'valid'
            elif gr == 'lung1-016':
                return 'test'
            else:
                return 'train'

        df = pd.DataFrame(data={'image_path': self.images_list, 'mask_path': self.masks_list})
        df['width'] = 512
        df['height'] = 512
        df['group'] = df['image_path'].apply(lambda x: x[self.pref_len:self.pref_len + self.lung_len])
        df['slice'] = df['image_path'].apply(lambda x: int(x.split('-')[2][:3]))
        df['split'] = df['group'].apply(lambda x: split(x))
        df.to_csv(self.paths['root dir'] + 'train.csv', index=False)
        return df

    def get_3d_masked_img_ani(self, img_3d_num, start_slice, stop_slice):
        test_images_path = self.images_3d_paths[img_3d_num]
        test_masks_path = self.masks_3d_paths[img_3d_num]
        test_images_3d = self.load_dicom(test_images_path)
        max_slice_num = test_images_3d.shape[0]
        test_masks_3d = nib.load(test_masks_path)
        test_masks_3d = test_masks_3d.get_fdata().transpose(2, 0, 1)

        fig, ax = plt.subplots()
        ims = []
        lung_label = test_images_path[self.pref_img_len:self.pref_img_len + self.lung_len]
        plt.legend([Rectangle((0, 0), 1, 1)], [lung_label])

        for slice_num in range(min(start_slice, max_slice_num), min(stop_slice, max_slice_num)):
            img, msk = test_images_3d[slice_num], test_masks_3d[slice_num]
            msk = cv2.rotate(msk, cv2.ROTATE_90_COUNTERCLOCKWISE)
            im1 = ax.imshow(img, cmap='bone', animated=True)
            im2 = ax.imshow(msk, alpha=0.4, animated=True)
            ims.append([im1, im2])

        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
        return ani

    def clear_lists(self):
        self.images_3d_paths.clear()
        self.masks_3d_paths.clear()
        self.images_list.clear()
        self.masks_list.clear()

    def run(self):
        self.clear_lists()
        self.get_3d_paths()
        self.save_png_slices()
        return self.make_metadata_frame()


def test_process_data():
    data_prep = DataPreprocessor(Config.paths)
    print(data_prep.run())


if __name__ == '__main__':
    test_process_data()
