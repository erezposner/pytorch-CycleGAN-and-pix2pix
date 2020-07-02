from pathlib import Path

import glob
import shutil

dataset_path = Path('/opt/data/DVP')
samples = glob.glob(str(dataset_path / '*'))
dataset_path = Path('./datasets/DVP')
import numpy as np

modes = ['train', 'test', 'val']
for mode in modes:
    output_A_folder = (dataset_path / 'A' / mode)
    output_B_folder = (dataset_path / 'B' / mode)
    output_A_folder.mkdir(parents=True, exist_ok=True)
    output_B_folder.mkdir(parents=True, exist_ok=True)
for sample in samples:
    mode = modes[np.random.randint(0, 3)]
    output_A_folder = (dataset_path / 'A' / mode)
    output_B_folder = (dataset_path / 'B' / mode)
    shutil.copy(str(Path(sample) / 'rendered.jpg'), str(Path(output_B_folder) / (Path(sample).stem + '.jpg')))
    shutil.copy(str(Path(sample) / 'original.jpg'), str(Path(output_A_folder) / (Path(sample).stem + '.jpg')))

    # print(mode)
