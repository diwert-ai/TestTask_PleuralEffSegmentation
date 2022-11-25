import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import nibabel as nib
import cv2
import SimpleITK as sitk


class DataPreprocessor:
    def __init__(self):
        self.paths = {'root dir': 'e:/train/',
                      'exp dir': 'e:/train/output/',
                      '3d images dir': 'e:/train/3d_images/',
                      '3d masks dir': 'e:/train/3d_masks/',
                      '3d images files': 'e:/train/3d_images/L*/*/*/*',
                      '3d masks files': 'e:/train/3d_masks/L*/*gz',
                      'images train': 'e:/train/data/',
                      'masks train': 'e:/train/data/masks/'}
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

    def run(self):
        self.get_3d_paths()
        self.save_png_slices()
        return self.make_metadata_frame()


def test_process_data():
    data_prep = DataPreprocessor()
    print(data_prep.run())


if __name__ == '__main__':
    test_process_data()
