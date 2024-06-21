import os
import json 
import pathlib
import argparse
import nibabel as nib
import numpy as np
from tqdm import tqdm
from PIL import Image


def z_score_normalize(array):
    array = array.astype(np.float32)
    mask = array > 0
    mean = np.mean(array[mask])
    std = np.std(array[mask])
    array -= mean
    array /= std
    return array


def preprocess_brats(config):
    src_dir_path = config['src_dir_path']
    dst_dir_path = config['dst_dir_path']
    dst_image_size = config['image_size']
    modalities = config['modalities']

    src_dir_path = pathlib.Path(src_dir_path)

    for patient_id in tqdm(os.listdir(os.path.join(src_dir_path))):
        patient_dir_path = src_dir_path / patient_id

        for modality in modalities:
            file_path = os.path.join(
                patient_dir_path,
                patient_id + '_' + modality['pattern'] + '.nii.gz'
            )
            nii_file = nib.load(file_path)
            series = nii_file.get_fdata()

            if modality['name'] == 'SEG':
                series = series.astype(np.int32)
                bincount = np.bincount(series.ravel())

                if 'Training' in config['src_dir_path'] and modality['pattern'] == 'seg':
                    if len(bincount) > 3:
                        assert bincount[3] == 0

                    series[series == 4] = 3  # 3: ET (GD-enhancing tumor)
                    series[series == 2] = 2  # 2: ED (peritumoral edema)
                    series[series == 1] = 1  # 1: NCR/NET (non-enhancing tumor core)
                    series[series == 0] = 0  # 0: Background

            else:
                series = z_score_normalize(series)

            for i in range(series.shape[2]):
                slice = series[..., i]
                slice = np.rot90(slice, k=3)

                if modality['name'] == 'SEG':
                    slice = np.array(Image.fromarray(slice.astype(np.uint8)).resize(
                        (dst_image_size, dst_image_size),
                        resample=Image.NEAREST,
                    ))

                else:
                    slice = np.array(Image.fromarray(slice).resize(
                        (dst_image_size, dst_image_size),
                        resample=Image.BILINEAR,
                    ))

                dst_patient_dir_path = os.path.join(
                    config['dst_dir_path'], patient_id
                )
                os.makedirs(dst_patient_dir_path, exist_ok=True)

                save_path = os.path.join(
                    dst_patient_dir_path,
                    patient_id + '_' + modality['save_pattern'] + '_' + str(i).zfill(4) + '.npy'
                )

                np.save(save_path, slice)


def main(config_path):
    with open(config_path) as f:
        config = json.load(f)

    for dataset_name, dataset_config in config.items():
        print(f'Preprocessing {dataset_name}...')
        preprocess_brats(dataset_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='preprocess_config.json',type=str)
    args = parser.parse_args()

    main(args.config)