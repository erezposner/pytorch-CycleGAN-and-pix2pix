import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import pickle
from pathlib import Path
from torchvision import transforms


def create_mask_from_white_background(A):
    import numpy as np
    import cv2
    # im = Image.new(mode="RGB", size=A.size)
    im = A.convert('RGBA')
    data = np.array(im)

    cv2.imwrite('out/t.png',
                data.astype(np.uint8))
    rgb = data

    # color = [246, 213, 139]  # Original value
    black = [0, 0, 0, 255]
    white_offset = [245, 245, 245, 255]
    white = [255, 255, 255, 255]
    mask = np.all(rgb >= white_offset, axis=-1)
    # # change all pixels that match color to white
    data[mask] = black
    data[~mask] = white
    data = data[...,0]
    # cv2.imwrite('out/t.png',
    #             data.astype(np.uint8))
    data = Image.fromarray(data)
    data.save('out/true_mask.png')
    return data
class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        with open(str(Path(AB_path).parent / f'{Path(AB_path).stem}.pkl'), 'rb') as f:
            metadata = pickle.load(f, encoding='latin1')
        silh_im = Image.fromarray(metadata['rendered_silh'])
        correspondence_map_im = Image.fromarray(metadata['correspondence_map'])
        normals_map_im = Image.fromarray(metadata['normals_map'])
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        true_mask = create_mask_from_white_background(A)
        B = AB.crop((w2, 0, w, h))
        if self.opt.constant_data:
            B = Image.open(r'bareteeth.000001.26_C/coma_2/mesh.png').convert('RGB')#TODO remove
        # C = AB.crop((2*w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        ext_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1),minus1To1=False)
        meta_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        # C_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        # from torchvision import transforms
        # transforms.ToPILImage()(A.cpu()).save('a.png')
        A = A_transform(A)
        B = B_transform(B)
        # C = C_transform(C)
        osize = [self.opt.load_size, self.opt.load_size]

        # trn = transforms.Compose([transforms.Resize(osize, Image.BICUBIC) ,
        #                   transforms.ToTensor()
        #                   ])
        silh_im = ext_transform(silh_im)
        true_mask = ext_transform(true_mask)
        correspondence_map_im = meta_transform(correspondence_map_im)
        normals_map_im = meta_transform(normals_map_im)
        # return {'A': A, 'B': B,'C': C, 'A_paths': AB_path, 'B_paths': AB_path, 'C_paths': AB_path}
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'true_flame_params': metadata['true_flame_params'],
                'silh':silh_im,'true_mask':true_mask,'normals_map_im':normals_map_im,'correspondence_map_im':correspondence_map_im}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
