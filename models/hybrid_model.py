from collections import OrderedDict

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, PointLights, TexturedSoftPhongShader, \
    look_at_view_transform, OpenGLPerspectiveCameras, RasterizationSettings, BlendParams, Materials, \
    SoftSilhouetteShader
from pytorch3d.renderer.mesh.shader import UVsCorrespondenceShader
from pytorch3d.structures import Textures
import numpy as np

from util.util import UnNormalize, Normalize
from .base_model import BaseModel
from . import networks
from .FlameDecoder import FlameDecoder
from types import SimpleNamespace
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path
from .face_part_seg.FacePartSegmentation import FacePartSegmentation

from .mesh_making import make_mesh
import pickle
import cv2

import torch.nn.functional as F


class HybridModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256_stub_318', dataset_mode='cg_aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=5 / 2 * 6.42 * 1e-5, help='weight for L1 loss')
            parser.add_argument('--lambda_silhouette', type=float, default=1e-5,
                                help='weight for silhouette loss')
            parser.add_argument('--lambda_face_seg', type=float, default=1.5 * 3.75e-5 / 1.28,
                                help='weight for face parts segmentation loss')
            parser.add_argument('--lambda_flame_regularizer', type=float, default=200.0,  # 500.0
                                help='weight for flame regularizer loss')

        return parser

    def get_additional_visuals(self):
        try:
            visual_ret = OrderedDict()
            visual_ret['vertices'] = self.vertices
            visual_ret['estimated_mesh'] = self.estimated_mesh
            visual_ret['estimated_texture_map'] = self.estimated_texture_map
            visual_ret['flamelayer'] = self.flamelayer
            visual_ret['true_mesh'] = self.true_mesh
        except:
            visual_ret = None
        return visual_ret

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        config = SimpleNamespace(batch_size=self.opt.batch_size, flame_model_path='./smpl_model/flame2020/male_model.pkl')

        self.backgrounds_folder = f'resources/backgrounds'
        self.num_of_backgrounds = len(list(Path(self.backgrounds_folder).glob('*')))
        self.flamelayer = FlameDecoder(config)
        self.flamelayer.to(self.device)

        # config.use_3D_translation = True  # could be removed, depending on the camera model
        # config.use_face_contour = False
        self.face_parts_segmentation = FacePartSegmentation(self.device)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', '3d_face_part_segmentation', 'segmented_ears_colored_variance']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'fake_Texture', 'fake_B', 'real_B','loss_G_L1_reducted']
        self.visual_names = ['real_A', 'warped_input_texture_map', 'fake_Texture', 'yam_rendered_img', 'fake_B', 'real_B', 'loss_3d_face_part_segmentation_de']
        # self.visual_names = ['fake_Texture', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'Global_shape']
        else:  # during test time, only load G
            self.model_names = ['G', 'Global_shape']

        # initialize flame parameters
        self.shape_params_size = 300
        self.expression_params_size = 100
        self.neck_pose_params_size = 3
        self.jaw_pose_size = 3
        self.global_rot_size = 3
        self.transl_size = 3
        self.eyball_pose_size = 6
        # define networks (both generator and discriminator)
        self.netGlobal_shape = networks.define_global_shape(self.shape_params_size, self.gpu_ids)
        # TODO changed to 5 to generate uvs as well

        # self.netG = networks.define_G(opt.input_nc * 2, opt.output_nc  , opt.ngf, opt.netG, opt.norm,
        self.netG = networks.define_G(opt.input_nc * 2, 5, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, image_size=self.opt.crop_size)
        self.netF = networks.define_F(opt.input_nc, opt.output_flame_params, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc * 2 + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            # self.visual_names.append('loss_G_L1_reducted')
            self.criterionGAN = networks.GANLoss(opt.gan_mode, soft_labels=self.opt.soft_labels).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction='none')
            # self.CrossEntropyCriterion = torch.nn.CrossEntropyLoss(reduction='none')
            self.CrossEntropyCriterion1 = torch.nn.NLLLoss(reduction='none')
            self.CrossEntropyCriterion2 = torch.nn.L1Loss(reduction='none')
            self.criterionBCE = torch.nn.BCELoss(reduction='none')

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(list(self.netF.parameters()) + list(self.netGlobal_shape.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.SGD(self.netD.parameters(), lr=opt.lr)  # TODO Check is SGD is better than ADAM
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)

        self.init_differential_renderer()
        self.set_default_weights()

    def init_differential_renderer(self):

        distance = 0.3
        R, T = look_at_view_transform(distance, 0, 0)
        cameras = OpenGLPerspectiveCameras(device=self.device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=self.opt.crop_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            cull_backfaces=True
        )
        silhouette_raster_settings = RasterizationSettings(
            image_size=self.opt.crop_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
        )
        # Change specular color to green and change material shininess
        self.materials = Materials(
            device=self.device,
            ambient_color=[[1.0, 1.0, 1.0]],
            specular_color=[[0.0, 0.0, 0.0]],
            diffuse_color=[[1.0, 1.0, 1.0]],
        )
        bp = BlendParams(background_color=(0, 0, 0))  # black
        # bp = BlendParams(background_color=(1, 1, 1))  # white is default

        lights = PointLights(device=self.device, location=((0.0, 0.0, 2.0),))

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),

            shader=TexturedSoftPhongShader(
                # blend_params=bp,
                device=self.device,
                lights=lights,
                cameras=cameras,
            )
        )

        # segmentation_texture_map = cv2.imread(str(Path('resources') / 'part_segmentation_map_2048_gray_n_h.png'))[...,
        segmentation_texture_map = cv2.imread(str(Path('resources') / 'Color_Map_Sag_symmetric.png'))[...,
                                   ::-1].astype(np.uint8)
        segmentation_texture_map = cv2.resize(segmentation_texture_map, (512, 512), interpolation=cv2.INTER_NEAREST)

        # import matplotlib.pyplot as plt
        # plt.imshow(segmentation_texture_map)
        # plt.show()

        segmentation_texture_map = (torch.from_numpy(np.array(segmentation_texture_map))).unsqueeze(0).float()
        self.segmentation_3d_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),

            shader=UVsCorrespondenceShader(
                blend_params=bp,
                device=self.device,
                cameras=cameras,
                colormap=segmentation_texture_map
            )
        )

        weights_texture_map = cv2.imread(str(Path('resources') / 'Color_Map_Sag_symmetric_Weights.png'))[...,
                              ::-1].astype(np.uint8)
        weights_texture_map = cv2.resize(weights_texture_map, (512, 512), interpolation=cv2.INTER_CUBIC)

        # import matplotlib.pyplot as plt
        # plt.imshow(weights_texture_map)
        # plt.show()

        weights_texture_map = (torch.from_numpy(np.array(weights_texture_map))).unsqueeze(0).float() / 255
        self.weights_3d_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),

            shader=UVsCorrespondenceShader(
                blend_params=bp,
                device=self.device,
                cameras=cameras,
                colormap=weights_texture_map
            )
        )

        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=silhouette_raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-10, gamma=1e-4))
        )
        self.negative_silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-10, gamma=1e-4))
        )
        self.texture_data = np.load('smpl_model/texture_data.npy', allow_pickle=True, encoding='latin1').item()
        self.verts_uvs1 = torch.tensor(self.texture_data['vt'], dtype=torch.float32).unsqueeze(0).cuda(
            self.device)
        self.faces_uvs1 = torch.tensor(self.texture_data['ft'].astype(np.int64), dtype=torch.int64).unsqueeze(0).cuda(
            self.device)

    def compute_visuals(self):
        # print('hybrid')

        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def set_input(self, input_data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input_data['A' if AtoB else 'B'].to(self.device)
        self.real_B = input_data['B' if AtoB else 'A'].to(self.device)
        self.true_flame_params = input_data['true_flame_params']
        self.yam_rendered_img = input_data['yam_rendered_img'].to(self.device)
        for k in self.true_flame_params.keys():
            self.true_flame_params[k] = self.true_flame_params[k].to(self.device)
        self.silh = input_data['silh'].to(self.device)
        self.true_mask = input_data['true_mask'].to(self.device)
        self.fake_normal_map = input_data['normals_map_im'].to(self.device)
        self.fake_correspondence_map = input_data['correspondence_map_im'].to(self.device)
        self.true_mask = self.true_mask.clamp(0, 1)
        self.image_paths = input_data['A_paths' if AtoB else 'B_paths']
        self.create_true_mesh_from_initial_guess()

    def create_true_mesh_from_initial_guess(self):

        self.true_vertices = self.flamelayer(shape_params=self.true_flame_params['shape_params'][0], expression_params=self.true_flame_params['expression_params'][0],
                                             pose_params=torch.cat([self.true_flame_params['global_rot'][0], self.true_flame_params['jaw_pose'][0]], dim=1),
                                             neck_pose=self.true_flame_params['neck_pose_params'][0],
                                             transl=self.true_flame_params['transl'][0],
                                             eye_pose=torch.zeros((1, self.eyball_pose_size)).cuda())
        texture_map = UnNormalize(self.real_A).permute(0, 2, 3, 1)
        texture = Textures(texture_map, faces_uvs=self.faces_uvs1, verts_uvs=self.verts_uvs1)
        #
        self.true_mesh = make_mesh(self.true_vertices.squeeze(), self.flamelayer.faces, False, texture)
        # final_obj = os.path.join('out/', 'final_model.obj')
        #
        # save_obj(final_obj, self.true_mesh.verts_packed(), torch.from_numpy(self.flamelayer.faces.astype(np.int32)),
        #          verts_uvs=self.true_mesh.textures.verts_uvs_packed(), texture_map=texture_map,
        #          faces_uvs=self.true_mesh.textures.faces_uvs_packed())

    def create_geo_from_flame_params(self, flame_param, base_flame_params=None, use_fix_params=False):
        # scale = 0.0001
        scale = 0.0001
        # scale = 10

        if base_flame_params is None:
            base_flame_params = defaultDict()
            base_flame_params['shape_params'] = torch.zeros((1, 1, self.shape_params_size)).cuda()
            base_flame_params['expression_params'] = torch.zeros((1, 1, self.expression_params_size)).cuda()
            base_flame_params['neck_pose_params'] = torch.zeros((1, 1, self.neck_pose_params_size)).cuda()
            base_flame_params['jaw_pose'] = torch.zeros((1, 1, self.jaw_pose_size)).cuda()
            base_flame_params['global_rot'] = torch.zeros((1, 1, self.global_rot_size)).cuda()
            base_flame_params['transl'] = torch.zeros((1, 1, self.transl_size)).cuda()

        if use_fix_params:
            flame_param = torch.zeros((1, 118)).cuda()
            self.shape_params = base_flame_params['shape_params']
        else:
            # self.shape_params = self.netGlobal_shape.module.global_shape + base_flame_params['shape_params']
            self.shape_params = base_flame_params['shape_params']  # TODO always use loaded shape

        # Creating a batch of mean shapes
        ind = 0
        # if use_fix_params:
        #     flame_param[:, ind:shape_params_size] = data['shape_params']
        # self.shape_params = flame_param[:, ind:self.shape_params_size] + base_flame_params['shape_params']

        # ind += self.shape_params_size
        # if use_fix_params:
        # flame_param[:, ind:ind + expression_params_size] = data['expression_params']
        self.expression_params = flame_param[:, ind:ind + self.expression_params_size] + \
                                 base_flame_params['expression_params'][0]
        # self.expression_params =  base_flame_params['expression_params'][0]
        ind += self.expression_params_size
        # if use_fix_params:
        # flame_param[:, ind:ind + neck_pose_params_size] = data['neck_pose_params']
        self.neck_pose = flame_param[:, ind:ind + self.neck_pose_params_size] + base_flame_params['neck_pose_params'][0]
        ind += self.neck_pose_params_size
        # if use_fix_params:
        #     flame_param[:, ind:ind + jaw_pose_size] = data['jaw_pose']
        self.jaw_pose = flame_param[:, ind:ind + self.jaw_pose_size] + base_flame_params['jaw_pose'][0]
        # self.jaw_pose =  base_flame_params['jaw_pose'][0]
        ind += self.jaw_pose_size
        # if use_fix_params:
        #     flame_param[:, ind:ind + global_rot_size] = data['global_rot']
        global_rot = flame_param[:, ind:ind + self.global_rot_size] * scale + base_flame_params['global_rot'][0]

        ind += self.global_rot_size

        self.pose_params = torch.cat([global_rot, self.jaw_pose], dim=1)
        # if use_fix_params:
        #     flame_param[:, ind:ind + transl_size] = data['transl']
        self.transl = flame_param[:, ind:ind + self.transl_size] * scale + base_flame_params['transl'][0]

        ind += self.transl_size
        self.eyball_pose = flame_param[:, ind:ind + self.eyball_pose_size]
        ind += self.eyball_pose_size
        vertices = self.flamelayer(shape_params=self.shape_params[0], expression_params=self.expression_params,
                                   pose_params=self.pose_params, neck_pose=self.neck_pose, transl=self.transl,
                                   eye_pose=self.eyball_pose)
        return vertices

    def project_to_image_plane(self, vertices, texture_map, use_constant_data):
        # self.renderer
        if use_constant_data:  # hardcoded example
            with torch.no_grad():
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])

                direc = Path('bareteeth.000001.26_C/coma_2')
                tex = Image.open(direc / 'mesh.png')
                texture_map = transform(tex).unsqueeze(0)
                mesh = load_objs_as_meshes([direc / 'mesh.obj'], device=self.device)
                vertices = mesh.verts_padded()
        # import datetime
        # now = datetime.datetime.now()
        # final_obj = f'{self.save_dir}/web/images/{now.strftime("%Y-%m-%d_%H:%M:%S")}_fake_mesh.obj'
        # final_obj = f'{self.save_dir}/web/images/{self.opt.epoch_count:03d}_fake_mesh.obj'
        # save_obj(final_obj, vertices[0], torch.from_numpy(self.flamelayer.faces.astype(np.int32)))
        self.estimated_texture_map = texture_map.permute(0, 2, 3, 1)
        texture = Textures(self.estimated_texture_map, faces_uvs=self.faces_uvs1, verts_uvs=self.verts_uvs1)
        self.vertices = vertices
        self.estimated_mesh = make_mesh(vertices.squeeze(), self.flamelayer.faces, False, texture)
        # final_obj = os.path.join('out/', 'final_model.obj')

        # save_obj(final_obj, self.estimated_mesh.verts_packed(), torch.from_numpy(self.flamelayer.faces.astype(np.int32)),
        #          verts_uvs=self.estimated_mesh.textures.verts_uvs_packed(), texture_map=self.estimated_texture_map,
        #          faces_uvs=self.estimated_mesh.textures.faces_uvs_packed())

        images = self.renderer(self.estimated_mesh, materials=self.materials)
        silhouette_images = self.silhouette_renderer(self.estimated_mesh, materials=self.materials)[..., 3].unsqueeze(0)
        negative_silhouette_images = self.negative_silhouette_renderer(self.estimated_mesh, materials=self.materials)[
            ..., 3].unsqueeze(0)
        if self.opt.verbose:
            transforms.ToPILImage()(silhouette_images.squeeze().permute(0, 1).cpu()).save('out/silhouette.png')
            # transforms.ToPILImage()(images.squeeze().permute(2, 0, 1).cpu()).save('out/img.png')
        cull_backfaces_mask = (1 - (silhouette_images - negative_silhouette_images).abs())
        if self.opt.verbose:
            img = (images[0][..., :3].detach().cpu().numpy() * 255).astype(np.uint8)

            Image.fromarray(img).save('out/test1.png')
        images = Normalize(images)
        silhouette_images = silhouette_images.clamp(0, 1)
        segmented_3d_model_image = self.segmentation_3d_renderer(self.estimated_mesh)[0, ..., 0].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        weights_3d_renderer_image = self.weights_3d_renderer(self.estimated_mesh)[0, ..., 0].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        # cv2.imwrite('out/s.png', segmented_3d_model_image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
        # Image.fromarray(
        #     ((255 * segmented_3d_model_image[0, ..., :3]).squeeze().detach().cpu().numpy().astype(np.uint8))).save(
        #     str('out/segmentatino_texture.png')
        # )
        return images[..., :3].permute(0, 3, 1, 2), silhouette_images, cull_backfaces_mask, segmented_3d_model_image, weights_3d_renderer_image  # [..., :3].permute(0, 3, 1, 2)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        inp = torch.cat((self.real_A, self.fake_correspondence_map), 1)

        if True:
            # inp = self.real_A
            # from torchsummary import summary
            # summary(self.netG, (6, 256, 256))
            self.uvs, self.fake_flame = self.netG(inp)  # RefinedTextureMap = G(TextureMap)
            flow_field = torch.tanh(self.uvs[..., 0:2, :, :])
            fake_Texture_gen = self.uvs[..., 2:, :, :]
            use_uvs = True
            if use_uvs:  # don't forget to set net output channels to 2
                # region use displacement with inital guess that is 0
                yv, xv = torch.meshgrid([torch.arange(0, self.real_A.shape[3]), torch.arange(0, self.real_A.shape[2])])
                xv = xv.unsqueeze(0).unsqueeze(0) / 127.5 - 1
                yv = yv.unsqueeze(0).unsqueeze(0) / 127.5 - 1
                uvs_init = torch.cat((xv, yv), 1).permute(0, 2, 3, 1).to(self.device)
                self.warped_input_texture_map = F.grid_sample(self.real_A, uvs_init + flow_field.permute(0, 2, 3, 1) / 10, align_corners=True)
                self.fake_Texture = fake_Texture_gen + self.warped_input_texture_map
                # self.fake_Texture = fake_Texture_gen + F.grid_sample(self.real_A, uvs_init ,align_corners=True)
                # self.fake_Texture = F.grid_sample(self.real_A, uvs_init + self.uvs.permute(0, 2, 3, 1) / 10)
            else:
                self.uvs, _ = self.netG(inp)  # RefinedTextureMap = G(TextureMap)
                self.fake_B = self.uvs[..., 2:, :, :]
            # cv2.imwrite('out/t.png',
            #             (255 * UnNormalize(self.fake_Texture.detach()).cpu().squeeze().permute(1, 2, 0).numpy()).astype(np.uint8))
            # endregion
            # aaa = 255 * (
            #         self.real_A * 0.5 + 0.5)
            # self.fake_B = self.project_to_image_plane(self.fake_geo_from_flame, aaa
            #                                           )
            # cv2.imwrite('out/t.png',
            #             (self.fake_B.detach().cpu().squeeze().permute(1, 2, 0).numpy()).astype(np.uint8))
            # torch.autograd.set_detect_anomaly(True)
            # self.fake_flame = self.netF(self.real_B)  # FlameParams = G(CapturedImg)
            if self.opt.constant_data:
                with open('bareteeth.000001.26_C/coma_2/flame_params.pkl', 'rb') as file:
                    data = pickle.load(file)
                    with torch.no_grad():
                        self.fake_flame = data

            zero_out_estimated_geomtery = False
            self.fake_geo_from_flame = self.create_geo_from_flame_params(self.fake_flame / 1000, base_flame_params=self.true_flame_params, use_fix_params=zero_out_estimated_geomtery)

            self.fake_B, self.fake_B_silhouette, self.cull_backfaces_mask, self.segmented_3d_model_image, self.weights_3d_model_image = self.project_to_image_plane(self.fake_geo_from_flame,
                                                                                                                                                                    UnNormalize(self.fake_Texture),
                                                                                                                                                                    self.opt.constant_data)
            # self.estimated_texture_map = UnNormalize(self.real_A).permute(0, 2, 3, 1) #TODO remove
            # self.fake_B, self.fake_B_silhouette, self.cull_backfaces_mask, self.segmented_3d_model_image = self.project_to_image_plane(self.fake_geo_from_flame, UnNormalize(self.real_A),
            #                                                                                                                            self.opt.constant_data)
        else:
            self.fake_B = self.netG(self.real_A)  # G(Texture)
        # with torch.no_grad(): #TODO check test or remove
        # self.fake_B = self.fake_B * self.cull_backfaces_mask
        # self.real_B = self.real_B * self.cull_backfaces_mask
        # eventually should produce self.fake_B

    def set_default_weights(self):

        self.weights = {}
        # Weight of the landmark distance term
        self.weights['lmk'] = 1.0

        # weights for different regularization terms
        self.weights['laplace'] = 100
        self.weights['euc_reg'] = 0.1

        # Weights for flame params regularizers
        self.weights['shape'] = 1e-3
        # Weight of the expression regularizer
        self.weights['expr'] = 1e-3
        # Weight of the neck pose (i.e. neck rotationh around the neck) regularizer
        self.weights['neck_pose'] = 100.0
        # Weight of the jaw pose (i.e. jaw rotation for opening the mouth) regularizer
        self.weights['jaw_pose'] = 1e-3

    def flame_regularizer_loss(self, vertices):
        # pose_params = torch.cat([self.global_rot, self.jaw_pose], dim=1)

        shape_params = self.shape_params
        flame_reg = self.weights['neck_pose'] * torch.sum(self.neck_pose ** 2) + \
                    self.weights['jaw_pose'] * torch.sum(self.jaw_pose ** 2) + \
                    self.weights['shape'] * torch.sum(shape_params ** 2) + \
                    self.weights['expr'] * torch.sum(self.expression_params ** 2)
        return flame_reg

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_correspondence_map, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.fake_correspondence_map, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_correspondence_map, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B

        # self.loss_G_L1_reducted = self.loss_G_L1_reducted.abs().mean(1).unsqueeze(0).clamp(0, 255)

        self.loss_F_Reg = self.flame_regularizer_loss(self.fake_geo_from_flame) * self.opt.lambda_flame_regularizer
        # silhouette loss
        self.loss_silhouette_de = self.rect_mask.squeeze()[0] * self.criterionBCE(
            self.rect_mask.squeeze()[0] * self.fake_B_silhouette.squeeze(),
            self.rect_mask.squeeze()[0] * self.true_mask.squeeze())

        eta = 0.01
        self.loss_silhouette = self.loss_silhouette_de.sum() * self.opt.lambda_silhouette
        self.segmented_3d_one_hot_model_image = torch.nn.functional.one_hot(
            (self.segmented_3d_model_image[:, 0, :, :]).long()).float().permute(0, 3, 1, 2)

        segmented_ears_mask = self.segmented_3d_one_hot_model_image[:, self.segmentation_parts_dict['l_ear'], :, :].clone()

        self.segmented_3d_one_hot_model_image[self.segmented_3d_one_hot_model_image == 0] = self.segmented_3d_one_hot_model_image[self.segmented_3d_one_hot_model_image == 0] + eta
        # self.weights_3d_model_image = self.weights_3d_model_image[:, 0, :, :]
        self.loss_3d_face_part_segmentation = self.CrossEntropyCriterion1(torch.log(self.segmented_3d_one_hot_model_image),
                                                                          self.real_B_seg.long())
        self.loss_3d_face_part_segmentation = self.loss_3d_face_part_segmentation  # * self.weights_3d_model_image[:,0,:,:]

        # Todo This code below is for removing l1 loss where differences with segmentation - not tested!
        self.loss_3d_face_part_segmentation_de = self.loss_3d_face_part_segmentation.clamp(0, 255).unsqueeze(0)
        # self.loss_3d_face_part_segmentation_de = 1 - self.loss_3d_face_part_segmentation_de / self.loss_3d_face_part_segmentation_de.max()
        # self.loss_3d_face_part_segmentation_de = 255- self.loss_3d_face_part_segmentation_de
        # self.loss_3d_face_part_segmentation = self.CrossEntropyCriterion2(self.segmented_3d_model_image[:, 0, :, :],
        #                                                                  self.real_B_seg.float()) * self.opt.lambda_face_seg
        # self.loss_3d_face_part_segmentation_de = self.loss_3d_face_part_segmentation.clamp(0,255).unsqueeze(0) / 255
        # transforms.ToPILImage()(self.loss_3d_face_part_segmentation_de.cpu().squeeze()).save('out/loss_3d_face_part_segmentation.png')
        if self.opt.verbose:
            transforms.ToPILImage()(self.loss_3d_face_part_segmentation_de.cpu().squeeze()).save('out/loss_3d_face_part_segmentation.png')
            transforms.ToPILImage()(self.weights_3d_model_image.cpu().squeeze()).save('out/weights_3d_model_image.png')

        self.loss_3d_face_part_segmentation = self.loss_3d_face_part_segmentation.sum() * self.opt.lambda_face_seg  # * self.opt.lambda_face_seg
        self.loss_G_L1_reducted = self.criterionL1(self.fake_B,
                                                   self.real_B) * self.rect_mask  # * (1 - self.weights_3d_model_image)  # * self.loss_3d_face_part_segmentation_de  # * self.l1_weight_mask # * self.cull_backfaces_mask

        ## region compute ear variance
        segmented_ears_colored_variance = (segmented_ears_mask * UnNormalize(self.fake_B))

        r_segmented_ears_colored_variance = segmented_ears_colored_variance[:, 0, :, :]
        g_segmented_ears_colored_variance = segmented_ears_colored_variance[:, 1, :, :]
        b_segmented_ears_colored_variance = segmented_ears_colored_variance[:, 2, :, :]

        r_segmented_ears_colored_variance = r_segmented_ears_colored_variance[r_segmented_ears_colored_variance > 0].var()
        g_segmented_ears_colored_variance = g_segmented_ears_colored_variance[g_segmented_ears_colored_variance > 0].var()
        b_segmented_ears_colored_variance = b_segmented_ears_colored_variance[b_segmented_ears_colored_variance > 0].var()

        self.loss_segmented_ears_colored_variance = (r_segmented_ears_colored_variance + g_segmented_ears_colored_variance + b_segmented_ears_colored_variance) * 100
        # endregion
        self.loss_G_L1 = self.loss_G_L1_reducted.abs().sum() * self.opt.lambda_L1
        # self.loss_2d_face_part_segmentation = self.CrossEntropyCriterion(self.fake_B_seg,
        #                                                                  self.real_B_seg.long()) * self.opt.lambda_face_seg
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + \
                      self.loss_G_L1 + \
                      self.loss_F_Reg + \
                      self.loss_3d_face_part_segmentation + \
                      self.loss_segmented_ears_colored_variance
        # + self.loss_silhouette

        if self.opt.verbose:
            transforms.ToPILImage()(self.loss_G_L1_reducted.detach().cpu().squeeze() / 10).save('out/L1_reducted.png')
            transforms.ToPILImage()(255 * self.loss_silhouette_de.cpu()).save('out/loss_silhouette.png')
            transforms.ToPILImage()(self.segmented_3d_model_image.cpu().squeeze() / 255).save('out/fake_B_seg_cat.png')
            transforms.ToPILImage()(self.real_B_seg.cpu().squeeze() / 255).save('out/real_B_seg_cat.png')
            transforms.ToPILImage()(UnNormalize(self.real_B).cpu().squeeze()).save('out/real_B.png')
            transforms.ToPILImage()(UnNormalize(self.fake_B).cpu().squeeze()).save('out/fake_B.png')
            transforms.ToPILImage()(segmented_ears_colored_variance.squeeze().detach().cpu()).save('out/segmented_ear_mask.png')

        self.loss_G.backward()

    def perform_face_part_segmentation(self, input_img, fname='seg'):
        self.segmentation_parts_dict = {'skin': 1,
                                        'r_eyebrow': 2,
                                        'l_eyebrow': 3,
                                        'r_eye': 4,
                                        'l_eye': 5,
                                        'r_ear': 7,
                                        'l_ear': 8,
                                        'nose': 10,
                                        'mouth': 11,
                                        'u_lip': 12,
                                        'l_lip': 13,
                                        'neck': 14,
                                        'cloth': 16,
                                        'hair': 17,
                                        'hat': 18,
                                        }
        real_un = UnNormalize(input_img).squeeze()
        tt = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        real_un = F.interpolate(real_un.unsqueeze(0), (512, 512))
        img = tt(real_un.squeeze()).unsqueeze(0)

        # with torch.no_grad():
        out = self.face_parts_segmentation(img)
        parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
        # print(parsing)

        if self.opt.verbose:
            image = transforms.ToPILImage()(UnNormalize(input_img).squeeze().detach().cpu()).resize((512, 512), Image.BILINEAR)
            self.face_parts_segmentation.vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=f'out/{fname}.png')
        out = F.interpolate(out, (self.opt.crop_size, self.opt.crop_size))
        argmax_tensor = torch.argmax(out, dim=1)

        argmax_tensor[argmax_tensor == self.segmentation_parts_dict['r_eye']] = self.segmentation_parts_dict['l_eye']
        argmax_tensor[argmax_tensor == self.segmentation_parts_dict['r_ear']] = self.segmentation_parts_dict['l_ear']
        argmax_tensor[argmax_tensor == self.segmentation_parts_dict['r_eyebrow']] = self.segmentation_parts_dict['l_eyebrow']
        argmax_tensor[argmax_tensor == self.segmentation_parts_dict['hat']] = self.segmentation_parts_dict['hair']

        return out, argmax_tensor

    def mask_outputs(self):
        self.rect_mask = torch.zeros(self.fake_B.shape).cuda()

        self.l1_weight_mask = np.zeros((self.fake_B.shape[0], 1, self.fake_B.shape[2], self.fake_B.shape[3]))
        self.rect_mask[..., 0:195, 40:200] = 1
        h_c = 195 // 2
        w_c = 40 + (200 - 40) // 2

        # w_c = 128
        # h_c = 128
        x = np.linspace(0, self.l1_weight_mask.shape[2], self.l1_weight_mask.shape[2])
        y = np.linspace(0, self.l1_weight_mask.shape[3], self.l1_weight_mask.shape[3])

        xv, yv = np.meshgrid(x - w_c, y - h_c)
        self.l1_weight_mask = 5 * np.sqrt(xv ** 2 + yv ** 2) / np.sqrt((w_c ** 2 + h_c ** 2))
        # cv2.imwrite('out/d.png', (self.l1_weight_mask  * 25).astype(np.uint8))
        self.l1_weight_mask = torch.from_numpy(self.l1_weight_mask).repeat(1, 3, 1, 1).cuda().float()

        _, self.real_B_seg = self.perform_face_part_segmentation(self.real_B, fname='real_B_seg')
        self.real_B_seg[self.real_B_seg == 11] = 0  # mask mouth
        self.real_B_seg[self.real_B_seg == 14] = 0  # mash cloth
        self.real_B_seg[self.real_B_seg == 16] = 0  # mask neck
        self.fake_B_seg, _ = self.perform_face_part_segmentation(self.fake_B, fname='fake_B_seg')
        self.fake_B_seg[self.fake_B_seg == 14] = 0  # mash cloth
        self.fake_B_seg[self.fake_B_seg == 16] = 0  # mask neck

        # self.fake_B = Normalize(UnNormalize(self.fake_B) * self.rect_mask)
        # self.real_B = Normalize(UnNormalize(self.real_B) * self.rect_mask)

        self.segmented_3d_model_image = self.segmented_3d_model_image * self.rect_mask[:, 0, ...]
        self.segmented_3d_model_image[self.segmented_3d_model_image.round() == self.segmentation_parts_dict['cloth']] = 0  # mash cloth 16
        self.segmented_3d_model_image[self.segmented_3d_model_image.round() == self.segmentation_parts_dict['neck']] = 0  # mask neck 14

        # mask cloth & neck for images
        mask_seg_val = 1

        self.fake_B = Normalize(UnNormalize(self.fake_B) * self.rect_mask)
        self.real_B = Normalize(UnNormalize(self.real_B) * self.rect_mask)
        self.real_B_seg = self.real_B_seg * self.rect_mask[:, 0, ...]
        self.fake_B_seg = self.fake_B_seg * self.rect_mask[:, 0, ...]
        self.fake_B[self.segmented_3d_model_image.sum(dim=1).repeat(1, 3, 1, 1) == 0] = mask_seg_val
        self.real_B[self.real_B_seg.repeat(1, 3, 1, 1) == 0] = mask_seg_val
        # with torch.no_grad(): #add background image to images
        background_ind = np.random.randint(0, self.num_of_backgrounds)
        try:
            background_img = transforms.ToTensor()(Image.open(f'{self.backgrounds_folder}/img_{background_ind}.png'))
        except:
            background_img = transforms.ToTensor()(Image.open(f'{self.backgrounds_folder}/img_{background_ind}.jpg'))
        self.real_B = torch.where(self.real_B == mask_seg_val, Normalize(background_img[None, :3, ...]).cuda(), self.real_B)
        self.fake_B = torch.where(self.fake_B == mask_seg_val, Normalize(background_img[None, :3, ...]).cuda(), self.fake_B)
        #
        # cv2.imwrite('out/dr.png', 255 * UnNormalize(self.real_B).detach().cpu().squeeze().permute(1, 2, 0).numpy())
        # cv2.imwrite('out/df.png', 255 * UnNormalize(self.fake_B).detach().cpu().squeeze().permute(1, 2, 0).numpy())

        # self.mask = self.mask * self.true_mask

    def optimize_parameters(self):
        iteration = 1
        # seed = np.random.randint(0, 100000, (1,))
        seed = np.random.randint(0, 100000, 1)
        for i in range(iteration):  # discriminator-generator balancing
            torch.manual_seed(seed)
            np.random.seed(seed)

            self.forward()  # compute fake images: G(Texture) and G(Flame)
            self.mask_outputs()
            # update D
            if i == iteration - 1:
                self.set_requires_grad(self.netD, True)  # enable backprop for D
                self.optimizer_D.zero_grad()  # set D's gradients to zero
                self.backward_D()  # calculate gradients for D
                self.optimizer_D.step()  # update D's weights
            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.optimizer_F.zero_grad()  # set F's gradients to zero
            self.backward_G()  # calculate graidents for G
            self.optimizer_G.step()  # udpate G's weights
            self.optimizer_F.step()  # udpate F's weights
