import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool
import pickle
from pathlib import Path

import glob


def load_flame_params(flame_path):
    with open(flame_path, 'rb') as f:
        flame_params = pickle.load(f, encoding='latin1')
        if not isinstance(flame_params, Dict):
            flame_params_dict = {}
            flame_params_dict['global_rot'] = flame_params[0]
            flame_params_dict['transl'] = flame_params[1]
            flame_params_dict['shape_params'] = flame_params[2]
            flame_params_dict['expression_params'] = flame_params[3]
            flame_params_dict['jaw_pose'] = flame_params[4]
            flame_params_dict['neck_pose'] = flame_params[5]
            flame_params = flame_params_dict
    return flame_params


from typing import Dict


def image_write(path_A, path_B, path_ABC):
    im_A = cv2.imread(path_A, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_B, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_ABC, im_AB)


# --fold_A
# /home/user3/repos/pytorch-CycleGAN-and-pix2pix/datasets/before_combine/DVP_RANNI/A
# --fold_B
# /home/user3/repos/pytorch-CycleGAN-and-pix2pix/datasets/before_combine/DVP_RANNI/B
# --fold_C
# /home/user3/repos/pytorch-CycleGAN-and-pix2pix/datasets/before_combine/DVP_RANNI/C
# --fold_ABC
# /home/user3/repos/pytorch-CycleGAN-and-pix2pix/datasets/DVP_RANNI
# --no_multiprocessing
dataset = 'Iphone_new_demo'
parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str,
                    default=f'/home/user3/repos/pytorch-CycleGAN-and-pix2pix/datasets/before_combine/{dataset}/A')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str,
                    default=f'/home/user3/repos/pytorch-CycleGAN-and-pix2pix/datasets/before_combine/{dataset}/B')
parser.add_argument('--fold_C', dest='fold_C', help='input directory for image C', type=str,
                    default=f'/home/user3/repos/pytorch-CycleGAN-and-pix2pix/datasets/before_combine/{dataset}/C')
parser.add_argument('--fold_ABC', dest='fold_AB', help='output directory', type=str, default=f'/home/user3/repos/pytorch-CycleGAN-and-pix2pix/datasets/{dataset}')
parser.add_argument('--use_metadata', default=1, help='use metadata', type=str)
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)', action='store_true')
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing',
                    help='If used, chooses single CPU execution instead of parallel execution', action='store_true',
                    default=False)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

splits = os.listdir(args.fold_A)

if not args.no_multiprocessing:
    pool = Pool()

for sp in splits:
    img_fold_A = os.path.join(args.fold_A, sp)
    img_fold_B = os.path.join(args.fold_B, sp)
    img_fold_C = os.path.join(args.fold_C, sp)
    img_list = sorted(os.listdir(img_fold_A))
    if args.use_AB:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_AB = os.path.join(args.fold_AB, sp)
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        if args.use_AB:
            name_B = name_A.replace('_A.', '_B.')
        else:
            name_B = name_A
        name_C = name_A
        path_B = os.path.join(img_fold_B, name_B)
        path_C = os.path.join(img_fold_C, name_C)
        if os.path.isfile(path_A) and os.path.isfile(path_B) and os.path.isfile(path_C):
            name_AB = name_A
            if args.use_AB:
                name_AB = name_AB.replace('_A.', '.')  # remove _A
            path_ABC = os.path.join(img_fold_AB, name_AB)
            if not args.no_multiprocessing:
                pool.apply_async(image_write, args=(path_A, path_B, path_C, path_ABC))
            else:
                im_A = cv2.imread(path_A, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_B = cv2.imread(path_B, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_C = cv2.imread(path_C, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                default_size = 256
                im_A = cv2.resize(im_A, (default_size, default_size))
                im_B = cv2.resize(im_B, (default_size, default_size))
                im_C = cv2.resize(im_C, (default_size, default_size))

                if im_A.shape[0] != im_C.shape[0]:
                    im_C = cv2.resize(im_C, (im_A.shape[0], im_A.shape[1]))
                im_ABC = np.concatenate([im_A, im_B, im_C], 1)
                # im_ABC = np.concatenate([im_A, im_C], 1)
                cv2.imwrite(path_ABC, im_ABC)
                if args.use_metadata:
                    im_sil = cv2.imread(str(Path(path_C).parent / f'{Path(path_C).stem}_rendered_silhouette.jpg'))
                    with open(str(Path(path_C).parent / f'{Path(path_C).stem}_cam_params.npy'), 'rb') as f:
                        cam_params = np.load(f, encoding='latin1').item()
                    flame_path = str(Path(path_C).parent / f'{Path(path_C).stem}_flame_params.pkl')
                    flame_params = load_flame_params(flame_path)

                    with open(str(Path(path_C).parent / f'{Path(path_C).stem}_normals_map.pkl'), 'rb') as f:
                        normals_map = pickle.load(f, encoding='latin1')['normals_map']
                    with open(str(Path(path_C).parent / f'{Path(path_C).stem}_correspondence_map.pkl'), 'rb') as f:
                        correspondence_map = pickle.load(f, encoding='latin1')['correspondence_map']
                    metadata = {'true_flame_params': flame_params,
                                'rendered_silh': im_sil,
                                'correspondence_map': correspondence_map,
                                'normals_map': normals_map,
                                'cam_params': cam_params,
                                }
                    outpath = Path(path_ABC).parent / (Path(path_ABC).stem + '.pkl')
                    with open(outpath, 'wb') as outfile:
                        pickle.dump(metadata, outfile)

if not args.no_multiprocessing:
    pool.close()
    pool.join()
