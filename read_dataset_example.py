from models.FlameDecoder import FlameDecoder
from types import SimpleNamespace
from pathlib import Path
import pickle
import torch
from pytorch3d.structures import Textures
from models.mesh_making import make_mesh
from pytorch3d.io import save_obj
from torchvision import transforms
from PIL import Image
import os
import numpy as np

config = SimpleNamespace(batch_size=1, flame_model_path='./smpl_model/male_model.pkl')

flamelayer = FlameDecoder(config)
flamelayer.cuda()
data_path = Path('bareteeth.000001.26_C/coma_2')
texture_data = np.load('smpl_model/texture_data.npy', allow_pickle=True, encoding='latin1').item()
verts_uvs1 = torch.tensor(texture_data['vt'], dtype=torch.float32).unsqueeze(0).cuda()
faces_uvs1 = torch.tensor(texture_data['ft'].astype(np.int64), dtype=torch.int64).unsqueeze(0).cuda()
texture_map = Image.open(str(data_path / 'mesh.png'))
texture_map = transforms.ToTensor()(texture_map).unsqueeze(0)
with open(str(data_path / 'flame_params.pkl'), 'rb') as file:
    data = pickle.load(file)
    fake_flame = data
    shape_params = fake_flame['shape_params']
    expression_params = fake_flame['expression_params']

    global_rot = fake_flame['global_rot']
    jaw_pose = fake_flame['jaw_pose']
    neck_pose = fake_flame['neck_pose_params']
    transl = fake_flame['transl']
    pose_params = torch.cat([global_rot, jaw_pose], dim=1)
    eyball_pose = fake_flame['eye_pose']
vertices = flamelayer(shape_params=shape_params, expression_params=expression_params,
                      pose_params=pose_params, neck_pose=neck_pose, transl=transl,
                      eye_pose=eyball_pose)

estimated_texture_map = texture_map.permute(0, 2, 3, 1)
texture = Textures(estimated_texture_map, faces_uvs=faces_uvs1, verts_uvs=verts_uvs1)

estimated_mesh = make_mesh(vertices.squeeze(), flamelayer.faces, False, texture)
final_obj = os.path.join('out/', 'final_model.obj')
save_obj(final_obj, estimated_mesh.verts_packed(), torch.from_numpy(flamelayer.faces.astype(np.int32)),
         verts_uvs=estimated_mesh.textures.verts_uvs_packed(), texture_map=estimated_texture_map,
         faces_uvs=estimated_mesh.textures.faces_uvs_packed())
