from pathlib import Path

import glob
import shutil


def get_all_samples_from_cg_ds(ds_path):
    paths = sorted(Path(ds_path).rglob("features"))
    # paths = sorted(Path(ds_path).rglob("camera_0/features"))
    return paths


# TODO bind volume of dataset to /opt/data in settings

dataset_path = Path('/opt/data/Subject 1')
# samples = glob.glob(str(dataset_path / '*'))
samples = get_all_samples_from_cg_ds(dataset_path)
dataset_output_path = Path('./datasets/before_combine/DVP_CP')
import numpy as np

modes = ['train', 'test', 'val']
for mode in modes:
    output_A_folder = (dataset_output_path / 'A' / mode)
    output_B_folder = (dataset_output_path / 'B' / mode)
    output_C_folder = (dataset_output_path / 'C' / mode)
    output_A_folder.mkdir(parents=True, exist_ok=True)
    output_B_folder.mkdir(parents=True, exist_ok=True)
    output_C_folder.mkdir(parents=True, exist_ok=True)
for ind, sample in enumerate(samples):
    mode = modes[np.random.randint(0, 3)]
    output_A_folder = (dataset_output_path / 'A' / mode)
    output_B_folder = (dataset_output_path / 'B' / mode)
    output_C_folder = (dataset_output_path / 'C' / mode)
    shutil.copy(str(Path(sample) / 'original.jpg'), str(Path(output_A_folder) / (f'{ind:04d}' + '.jpg')))
    shutil.copy(str(Path(sample) / 'rendered.jpg'), str(Path(output_B_folder) / (f'{ind:04d}' + '.jpg')))
    shutil.copy(str(Path(sample) / 'mesh.png'), str(Path(output_C_folder) / (f'{ind:04d}' + '.jpg')))
    shutil.copy(str(Path(sample) / 'flame_params.pkl'),
                str(Path(output_C_folder) / (f'{ind:04d}' + '_flame_params.pkl')))
    shutil.copy(str(Path(sample) / 'silh.jpg'), str(Path(output_C_folder) / (f'{ind:04d}' + '_silh.jpg')))
    # shutil.copy(str(Path(sample) / 'original.jpg'), str(Path(output_A_folder) / (f'{ind:04d}_'+Path(sample).stem + '.jpg')))
    # shutil.copy(str(Path(sample) / 'rendered.jpg'), str(Path(output_B_folder) / (f'{ind:04d}_'+Path(sample).stem + '.jpg')))
    # shutil.copy(str(Path(sample) / 'mesh.png'), str(Path(output_C_folder) / (f'{ind:04d}_'+Path(sample).stem + '.jpg')))
    # shutil.copy(str(Path(sample) / 'flame_params.pkl'), str(Path(output_C_folder) / (f'{ind:04d}_'+Path(sample).stem + '_flame_params.pkl')))
    # shutil.copy(str(Path(sample) / 'silh.jpg'), str(Path(output_C_folder) / (f'{ind:04d}_'+Path(sample).stem + '_silh.jpg')))

    # print(mode)
