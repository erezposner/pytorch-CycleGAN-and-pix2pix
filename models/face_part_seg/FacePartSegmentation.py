import torch.nn as nn
import torch
from .model import BiSeNet
import cv2
import numpy as np


class FacePartSegmentation(nn.Module):

    def __init__(self, device='cuda'):
        super(FacePartSegmentation, self).__init__()
        n_classes = 19

        self.net = BiSeNet(n_classes=n_classes)
        self.net.to(device)
        save_pth = './models/face_part_seg/res/cp/79999_iter.pth'
        self.net.load_state_dict(torch.load(save_pth))
        self.net.eval()
    def forward(self, image):
        out = self.net(image)[0]
        # parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # # print(parsing)
        # print(np.unique(parsing))
        # image = transforms.ToPILImage()(self.UnNormalize(real).squeeze().detach().cpu()).resize((512, 512),
        #                                                                                         Image.BILINEAR)
        # vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path='out/seg.png')
        return out
    def vis_parsing_maps(self, im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
        # Colors for all 20 parts
        part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 0, 85], [255, 0, 170],
                       [0, 255, 0], [85, 255, 0], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]

        im = np.array(im)
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        # print(vis_parsing_anno_color.shape, vis_im.shape)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

        # Save result or not
        if save_im:
            cv2.imwrite(save_path[:-4] + '.png', vis_parsing_anno)
            cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # return vis_im
