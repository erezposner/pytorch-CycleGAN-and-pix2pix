from pathlib import Path

import glob
import shutil


def get_all_samples_from_cg_ds(ds_path):
    paths = sorted(Path(ds_path).rglob("features"))
    return paths


# TODO bind volume of dataset to /opt/data in settings

dataset_path = Path('/opt/data/Subject 1')
dataset_path = Path('/opt/data/master_reconstruction')
# samples = glob.glob(str(dataset_path / '*'))
samples = get_all_samples_from_cg_ds(dataset_path)
print(f'{len(samples)} were found. Starting copying...')
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
    # mode = modes[np.random.randint(0, 3)]
    sampled_mode_index = np.random.choice(a=[0, 1, 2], p=[0.7, 0.15, 0.15])
    mode = modes[sampled_mode_index]
    output_A_folder = (dataset_output_path / 'A' / mode)
    output_B_folder = (dataset_output_path / 'B' / mode)
    output_C_folder = (dataset_output_path / 'C' / mode)
    shutil.copy(str(Path(sample) / 'original.jpg'), str(Path(output_A_folder) / (f'{ind:04d}' + '.jpg')))
    shutil.copy(str(Path(sample) / 'rendered.jpg'), str(Path(output_B_folder) / (f'{ind:04d}' + '.jpg')))
    shutil.copy(str(Path(sample) / 'mesh.png'), str(Path(output_C_folder) / (f'{ind:04d}' + '.jpg')))
    shutil.copy(str(Path(sample) / 'flame_params.pkl'),
                str(Path(output_C_folder) / (f'{ind:04d}' + '_flame_params.pkl')))
    shutil.copy(str(Path(sample) / 'normals_map.p'),
                str(Path(output_C_folder) / (f'{ind:04d}' + '_normals_map.pkl')))
    shutil.copy(str(Path(sample) / 'correspondence_map.p'),
                str(Path(output_C_folder) / (f'{ind:04d}' + '_correspondence_map.pkl')))
    shutil.copy(str(Path(sample) / 'rendered_silhouette.jpg'), str(Path(output_C_folder) / (f'{ind:04d}' + '_rendered_silhouette.jpg')))

# import os

# # os.chdir('datasets')
# os.system(
#     'python datasets/combine_A_and_B_and_C_metadata.py --fold_A /home/user3/repos/pytorch-CycleGAN-and-pix2pix/datasets/before_combine/DVP_CP/A --fold_B /home/user3/repos/pytorch-CycleGAN-and-pix2pix/datasets/before_combine/DVP_CP/B --fold_C /home/user3/repos/pytorch-CycleGAN-and-pix2pix/datasets/before_combine/DVP_CP/C --fold_ABC /home/user3/repos/pytorch-CycleGAN-and-pix2pix/datasets/DVP_ABC_CP --no_multiprocessing')
