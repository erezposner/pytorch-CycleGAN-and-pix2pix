import numpy as np
import torch
import torch.nn as nn
import pickle
# from utils.laplacian import *
from smplx.lbs import lbs, batch_rodrigues, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords
from smplx.utils import Struct, to_tensor, to_np, rot_mat_to_euler


class FlameDecoder(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    """

    def __init__(self, config):
        super(FlameDecoder, self).__init__()
        print("Initializing a Flame decoder")
        with open(config.flame_model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        self.dtype = torch.float32
        self.batch_size = config.batch_size
        self.faces = self.flame_model.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        # Eyeball and neck rotation
        default_eyball_pose = torch.zeros((self.batch_size, 6),
                                          dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                         requires_grad=False))

        # Fixing 3D translation since we use translation in the image plane
        # self.use_3D_translation = config.use_3D_translation

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(self.flame_model.v_template),
                                       dtype=self.dtype))

        # The shape components
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(to_np(
            self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))
    def extract_vertices_uv_correspondence_for_tb(self,estimated_mesh,estimated_texture_map):
        # vertices_tensor = estimated_mesh.verts_padded()


        faces = self.faces.astype(np.int32).flatten()
        faces_uvs_packed = estimated_mesh.textures.faces_uvs_packed().cpu().numpy().flatten()
        zipped = np.concatenate((faces[:, None], faces_uvs_packed[:, None]), axis=1)

        vertices_uv_correspondence = np.unique(zipped, axis=0)

        return vertices_uv_correspondence
    def forward(self, shape_params=None, expression_params=None, pose_params=None, neck_pose=None, transl=None,
                eye_pose=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters
            return:
                vertices: N X V X 3
        """
        betas = torch.cat([shape_params, expression_params], dim=1)

        # If we don't specify eye_pose use the default
        eye_pose = (eye_pose if eye_pose is not None else self.eye_pose)

        full_pose = torch.cat([pose_params[:, :3], neck_pose, pose_params[:, 3:], eye_pose], dim=1)


        template_vertices = self.v_template.unsqueeze(0).repeat(pose_params.shape[0], 1, 1)
        vertices, _ = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)

        vertices += transl.unsqueeze(dim=1)

        return vertices
