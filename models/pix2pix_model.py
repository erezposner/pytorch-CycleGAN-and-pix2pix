import torch
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, PointLights, TexturedSoftPhongShader, \
    look_at_view_transform, OpenGLPerspectiveCameras, RasterizationSettings, BlendParams, DirectionalLights, Materials, \
    SoftSilhouetteShader
from pytorch3d.structures import Textures
import numpy as np
from .base_model import BaseModel
from . import networks
from .FlameDecoder import FlameDecoder
from types import SimpleNamespace
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path

from .face_part_seg.FacePartSegmentation import FacePartSegmentation
# from .face_part_seg.model import BiSeNet
# from .face_part_seg.test_face_seg import vis_parsing_maps
from .mesh_making import make_mesh
import pickle
import cv2
import os
import torch.nn.functional as F


class Pix2PixModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_silhouette', type=float, default=1e-5,
                                help='weight for silhouette loss')
            parser.add_argument('--lambda_face_seg', type=float, default=1,
                                help='weight for face parts segmentation loss')
            parser.add_argument('--lambda_flame_regularizer', type=float, default=500.0,  # 100.0
                                help='weight for flame regularizer loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        config = SimpleNamespace(batch_size=1, flame_model_path='./smpl_model/male_model.pkl')

        self.flamelayer = FlameDecoder(config)
        self.flamelayer.cuda()
        # config.use_3D_translation = True  # could be removed, depending on the camera model
        # config.use_face_contour = False
        self.face_parts_segmentation = FacePartSegmentation(self.device)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'silhouette', 'face_part_segmentation']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'fake_Texture', 'fake_B', 'real_B','loss_G_L1_reducted']
        self.visual_names = ['fake_Texture', 'fake_B', 'real_B', 'loss_G_L1_reducted']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # initialize flame parameters
        self.shape_params_size = 300
        self.expression_params_size = 100
        self.neck_pose_params_size = 3
        self.jaw_pose_size = 3
        self.global_rot_size = 3
        self.transl_size = 3
        self.eyball_pose_size = 6
        # define networks (both generator and discriminator)
        # TODO  opt.output_nc instead of 2 if generating images
        self.netG = networks.define_G(opt.input_nc, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netF = networks.define_F(opt.input_nc, opt.output_flame_params, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, soft_labels=self.opt.soft_labels).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction='none')
            self.CrossEntropyCriterion = torch.nn.CrossEntropyLoss()

            self.criterionBCE = torch.nn.BCELoss(reduction='none')

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.global_shape = torch.nn.Parameter(torch.randn((1, 1, self.shape_params_size)).to(self.device),
                                                   requires_grad=True)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(list(self.netF.parameters()) + [self.global_shape], lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)

        self.initDifferentialRenderer()
        self.set_default_weights()

    def initDifferentialRenderer(self):

        distance = 0.3
        R, T = look_at_view_transform(distance, 0, 0)
        cameras = OpenGLPerspectiveCameras(device=self.device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            cull_backfaces=True
        )
        silhouette_raster_settings = RasterizationSettings(
            image_size=256,
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
            # shininess=0
        )
        # bp = BlendParams(background_color=(-1, -1, -1))  # black
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

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.true_flame_params = input['true_flame_params']
        self.silh = input['silh'].to(self.device)
        self.true_mask = input['true_mask'].to(self.device)

        self.true_mask = self.true_mask.clamp(0, 1)

        # self.real_C = input['C'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def create_geo_from_flame_params(self, flame_param, base_flame_params=None, use_fix_params=False):
        scale = 0.0001
        # scale = 10

        if base_flame_params is None:
            base_flame_params = {}
            base_flame_params['shape_params'] = torch.zeros((1, 1, self.shape_params_size)).cuda()
            base_flame_params['expression_params'] = torch.zeros((1, 1, self.expression_params_size)).cuda()
            base_flame_params['neck_pose_params'] = torch.zeros((1, 1, self.neck_pose_params_size)).cuda()
            base_flame_params['jaw_pose'] = torch.zeros((1, 1, self.jaw_pose_size)).cuda()
            base_flame_params['global_rot'] = torch.zeros((1, 1, self.global_rot_size)).cuda()
            base_flame_params['transl'] = torch.zeros((1, 1, self.transl_size)).cuda()

        if use_fix_params:
            flame_param = torch.zeros((1, 418)).cuda()
        # Creating a batch of mean shapes
        ind = 0
        # if use_fix_params:
        #     flame_param[:, ind:shape_params_size] = data['shape_params']
        self.shape_params = flame_param[:, ind:self.shape_params_size] + base_flame_params['shape_params'][0]
        ind += self.shape_params_size
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
        # global_rot = global_rot.clamp(-1, 1)  # TODO check clamp rotation

        ind += self.global_rot_size

        self.pose_params = torch.cat([global_rot, self.jaw_pose], dim=1)
        # if use_fix_params:
        #     flame_param[:, ind:ind + transl_size] = data['transl']
        self.transl = flame_param[:, ind:ind + self.transl_size] * scale + base_flame_params['transl'][0]

        # self.transl = self.transl.clamp(-.3, .3)  # TODO check clamp translation
        ind += self.transl_size
        self.eyball_pose = flame_param[:, ind:ind + self.eyball_pose_size]
        ind += self.eyball_pose_size
        vertices = self.flamelayer(shape_params=self.global_shape[0], expression_params=self.expression_params,
                                   # vertices = self.flamelayer(shape_params=self.shape_params, expression_params=self.expression_params,
                                   pose_params=self.pose_params, neck_pose=self.neck_pose, transl=self.transl,
                                   eye_pose=self.eyball_pose)
        return vertices

    def project_to_image_plane(self, vertices, texture_map):
        # self.renderer
        if False:  # hardcoded example
            with torch.no_grad():
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])

                direc = Path('bareteeth.000001.26_C/minibatch_0_Netural_0')
                tex = Image.open(direc / 'mesh.png')
                texture_map = transform(tex).unsqueeze(0)
                mesh = load_objs_as_meshes([direc / 'mesh.obj'], device=self.device)
                vertices = mesh.verts_padded()
        # final_obj = os.path.join('out/', 'final_model.obj')
        import datetime
        now = datetime.datetime.now()
        final_obj = f'{self.save_dir}/web/images/{now.strftime("%Y-%m-%d_%H:%M:%S")}_fake_mesh.obj'
        # final_obj = f'{self.save_dir}/web/images/{self.opt.epoch_count:03d}_fake_mesh.obj'
        # save_obj(final_obj, vertices[0], torch.from_numpy(self.flamelayer.faces.astype(np.int32)))
        self.estimated_texture_map = texture_map.permute(0, 2, 3, 1)
        texture = Textures(self.estimated_texture_map, faces_uvs=self.faces_uvs1, verts_uvs=self.verts_uvs1)

        self.estimated_mesh = make_mesh(vertices.squeeze(), self.flamelayer.faces, False, texture)
        # save_obj(final_obj, estimated_mesh.verts_packed(), torch.from_numpy(self.flamelayer.faces.astype(np.int32)),
        #          verts_uvs=estimated_mesh.textures.verts_uvs_packed(), texture_map=self.estimated_texture_map,
        #          faces_uvs=estimated_mesh.textures.faces_uvs_packed())

        images = self.renderer(self.estimated_mesh, materials=self.materials)
        silhouette_images = self.silhouette_renderer(self.estimated_mesh, materials=self.materials)[..., 3].unsqueeze(0)
        negative_silhouette_images = self.negative_silhouette_renderer(self.estimated_mesh, materials=self.materials)[..., 3].unsqueeze(0)
        transforms.ToPILImage()(silhouette_images
                                .squeeze().permute(0, 1).cpu()).save('out/silhouette.png')
        # transforms.ToPILImage()(images
        #                         .squeeze().permute(2, 0, 1).cpu()).save('out/img.png')
        cull_backfaces_mask =  (1-(silhouette_images- negative_silhouette_images).abs())
        img = (images[0][..., :3].detach().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save('out/test1.png')
        images = self.Normalize(images)
        silhouette_images = silhouette_images.clamp(0, 1)

        return images[..., :3].permute(0, 3, 1, 2), silhouette_images,cull_backfaces_mask

    def UnNormalize(self, img):
        return img * 0.5 + 0.5

    def Normalize(self, img):
        return (img - 0.5) / 0.5

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if True:
            self.uvs = self.netG(self.real_A)  # RefinedTextureMap = G(TextureMap)
            use_uvs = False
            if use_uvs:  # don't forget to set net output channels to 2
                # region use displacement with inital guess that is 0
                yv, xv = torch.meshgrid([torch.arange(0, self.real_A.shape[3]), torch.arange(0, self.real_A.shape[2])])
                xv = xv.unsqueeze(0).unsqueeze(0) / 127.5 - 1
                yv = yv.unsqueeze(0).unsqueeze(0) / 127.5 - 1
                uvs_init = torch.cat((xv, yv), 1).permute(0, 2, 3, 1).to(self.device)
                self.fake_Texture = F.grid_sample(self.real_A, uvs_init + self.uvs.permute(0, 2, 3, 1) / 10)
            else:
                self.fake_Texture = self.uvs
            # endregion
            # aaa = 255 * (
            #         self.real_A * 0.5 + 0.5)
            # self.fake_B = self.project_to_image_plane(self.fake_geo_from_flame, aaa
            #                                           )
            # cv2.imwrite('out/t.png',
            #             (self.fake_B.detach().cpu().squeeze().permute(1, 2, 0).numpy()).astype(np.uint8))
            # torch.autograd.set_detect_anomaly(True)
            self.fake_flame = self.netF(self.real_B)  # FlameParams = G(CapturedImg)
            if self.opt.constant_data:
                with open('bareteeth.000001.26_C/coma_2/flame_params.pkl', 'rb') as file:
                    data = pickle.load(file)
                    with torch.no_grad():
                        self.fake_flame = data

            zero_out_estimated_texture_map = False
            self.fake_geo_from_flame = self.create_geo_from_flame_params(self.fake_flame,
                                                                         base_flame_params=self.true_flame_params,
                                                                         use_fix_params=zero_out_estimated_texture_map)

            # self.fake_B = self.project_to_image_plane(self.fake_geo_from_flame, self.UnNormalize(self.real_A))
            self.fake_B, self.fake_B_silhouette ,self.cull_backfaces_mask= self.project_to_image_plane(self.fake_geo_from_flame,
                                                                              self.UnNormalize(self.fake_Texture))
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
        flame_reg = self.weights['neck_pose'] * torch.sum(self.neck_pose ** 2) + self.weights[
            'jaw_pose'] * torch.sum(
            self.jaw_pose ** 2) + \
                    self.weights['shape'] * torch.sum(shape_params ** 2) + self.weights['expr'] * torch.sum(
            self.expression_params ** 2)
        return flame_reg

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B

        self.loss_G_L1_reducted = self.criterionL1(self.fake_B, self.real_B) * self.rect_mask #* self.cull_backfaces_mask
        self.loss_G_L1 = self.loss_G_L1_reducted.abs().mean() * self.opt.lambda_L1
        # image_pil = transforms.ToPILImage()(self.loss_G_L1[0].cpu())
        # image_pil.save('out/s.png')

        self.loss_F_Reg = self.flame_regularizer_loss(self.fake_geo_from_flame) * self.opt.lambda_flame_regularizer
        # silhouette loss
        self.loss_silhouette = self.rect_mask.squeeze()[0] * self.criterionBCE(
            self.rect_mask.squeeze()[0] * self.fake_B_silhouette.squeeze(),
            self.rect_mask.squeeze()[0] * self.true_mask.squeeze())
        image_pil = transforms.ToPILImage()(255 * self.loss_silhouette.cpu())
        image_pil.save('out/loss_silhouette.png')
        self.loss_silhouette = self.loss_silhouette.sum() * self.opt.lambda_silhouette

        self.loss_face_part_segmentation = self.CrossEntropyCriterion(self.fake_B_seg,
                                                                      self.real_B_seg.long()) * self.opt.lambda_face_seg

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_F_Reg   + self.loss_silhouette  # + self.loss_face_part_segmentation
        # self.loss_silhouette = self.loss_silhouette.repeat(3, 1, 1).permute(1,2,0)
        self.loss_G.backward()

    def perform_face_part_segmentation(self, input_img, fname='seg'):

        real_un = self.UnNormalize(input_img).squeeze()
        tt = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        real_un = F.interpolate(real_un.unsqueeze(0), (512, 512))
        img = tt(real_un.squeeze()).unsqueeze(0)

        # with torch.no_grad():
        out = self.face_parts_segmentation(img)
        parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
        # print(parsing)
        visualize = True  # TODO
        if visualize:
            # print(np.unique(parsing))
            image = transforms.ToPILImage()(self.UnNormalize(input_img).squeeze().detach().cpu()).resize((512, 512),
                                                                                                         Image.BILINEAR)
            self.face_parts_segmentation.vis_parsing_maps(image, parsing, stride=1, save_im=True,
                                                          save_path=f'out/{fname}.png')
        out = F.interpolate(out, (256, 256))
        parsing_tensor = torch.argmax(out, dim=1)

        return out, parsing_tensor

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(Texture) and G(Flame)
        # update D
        self.rect_mask = torch.zeros(self.fake_B.shape).cuda()
        self.rect_mask[..., 0:195, 40:200] = 1
        _, self.real_B_seg = self.perform_face_part_segmentation(self.real_B, fname='real_B_seg')
        self.fake_B_seg, _ = self.perform_face_part_segmentation(self.fake_B, fname='fake_B_seg')

        self.fake_B = self.Normalize(self.UnNormalize(self.fake_B) * self.rect_mask)
        self.real_B = self.Normalize(self.UnNormalize(self.real_B) * self.rect_mask)

        self.real_B_seg = self.real_B_seg * self.rect_mask[:, 0, ...]
        self.fake_B_seg = self.fake_B_seg * self.rect_mask[:, 0, ...]
        # self.mask = self.mask * self.true_mask

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
