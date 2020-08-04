from collections import defaultdict
import torch

import numpy as np
import os
import sys
import ntpath
import time

from torch.utils.tensorboard import SummaryWriter

from . import util, html
from subprocess import Popen, PIPE
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class TensorBoardVisualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """

        # self.writer.add_scalar("Loss/train", 10, 10)
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        existing_folders = len(list(Path(opt.logs_folder).glob(f'*{self.name}*')))
        self.writer = SummaryWriter(f'{opt.logs_folder}/{self.name}_exp_{existing_folders}',flush_secs=60)
        print(f'Visualizer Tensorboard is exporting to folder {opt.logs_folder}/{self.name}_exp_{existing_folders} . . .')
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            # import visdom
            self.ncols = opt.display_ncols
            # self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            # if not self.vis.check_connection():
            #     self.create_visdom_connections()

            if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
                self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
                self.img_dir = os.path.join(self.web_dir, 'images')
                print('create web directory %s...' % self.web_dir)
                util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False


    def display_estimated_mesh(self, epoch, flamelayer, estimated_mesh, estimated_texture_map, tag='mesh'):
        vertices_tensor = estimated_mesh.verts_padded()
        faces_tensor = torch.tensor(np.int32(flamelayer.faces), dtype=torch.long).cuda().unsqueeze(0)

        colors_tensor = torch.zeros(vertices_tensor.shape)

        verts_uvs =  estimated_mesh.textures.verts_uvs_packed().clone()
        verts_uvs[:, 1] = 1 - verts_uvs[:, 1] # invert horizontal axis



        verts_uvs_un = (verts_uvs * estimated_texture_map.shape[1] - 1).long()
        vertices_uv_correspondence = flamelayer.extract_vertices_uv_correspondence_for_tb(estimated_mesh, estimated_texture_map)
        for i in range(vertices_uv_correspondence.shape[0]):
            colors_tensor[0, vertices_uv_correspondence[i, 0], :] = estimated_texture_map[0, verts_uvs_un[vertices_uv_correspondence[i, 1], 1],
                                                                    verts_uvs_un[vertices_uv_correspondence[i, 1], 0],
                                                                    :].float() * 255

        self.writer.add_mesh(tag, vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor, global_step=epoch)




    def display_current_results(self, visuals, epoch, save_result, additional_visuals=None):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        try:
            self.display_estimated_mesh(epoch, additional_visuals['flamelayer'], additional_visuals['true_mesh'], additional_visuals['true_mesh'].textures.maps_padded(), 'true_mesh')

            self.display_estimated_mesh(epoch, additional_visuals['flamelayer'], additional_visuals['estimated_mesh'], additional_visuals['estimated_texture_map'], 'estimated_mesh')
        except:
            pass

        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            f = plt.figure(figsize=(15, 15))
            import math
            rows = int(math.ceil(len(visuals) / ncols))
            if ncols > 0:  # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                # table_css = """<style>
                #         table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                #         table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                #         </style>""" % (w, h)  # create a table css
                # # create a table of images.
                title = self.name
                # label_html = ''
                # label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    # label_html_row += '<td>%s</td>' % label
                    a = f.add_subplot(rows, ncols, idx + 1)
                    a.set_title(f'{label}')
                    plt.imshow(image_numpy, cmap="jet")

                    # Image.fromarray(image_numpy).save(f'out/{label}.png')
                    # images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    # if idx % ncols == 0:
                    #     label_html += '<tr>%s</tr>' % label_html_row
                    #     label_html_row = ''
                # white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                self.writer.add_figure('phase', f, epoch)

                # while idx % ncols != 0:
                # images.append(white_image)
                # label_html_row += '<td></td>'
                # idx += 1
                # if label_html_row != '':
                #     label_html += '<tr>%s</tr>' % label_html_row
                # try:
                #     self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                #                     padding=2, opts=dict(title=title + ' images'))
                #     label_html = '<table>%s</table>' % label_html
                #     self.vis.text(table_css + label_html, win=self.display_id + 2,
                #                   opts=dict(title=title + ' labels'))
                # except VisdomExceptionBase:
                #     self.create_visdom_connections()

            else:  # show each image in a separate visdom panel;
                idx = 1
                # try:
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    a = f.add_subplot(rows, ncols, idx)
                    a.set_title(f'{label}')
                    plt.imshow(image_numpy, cmap="jet")
                    idx += 1
                # except VisdomExceptionBase:
                #     self.create_visdom_connections()

            plt.close('all')
        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        d = defaultdict()

        for k in self.plot_data['legend']:
            d[f'{k}'] = torch.from_numpy(np.array(losses[k]))
            # self.writer.add_scalar(f'data/{k}', torch.from_numpy(np.array(losses[k])), self.plot_data['X'][-1])
        # print(self.plot_data['X'][-1])
        self.writer.add_scalars('data/scalar_group', d, self.plot_data['X'][-1])

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
