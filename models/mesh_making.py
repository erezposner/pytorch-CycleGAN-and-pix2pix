import torch
import numpy as np
from pytorch3d.structures import Meshes, Textures


def make_mesh(verts: torch.tensor, faces: np.ndarray, detach:bool, textures = None) -> Meshes:
    device = torch.device("cuda:0")
    if detach:
        verts = verts.detach()
    # Initialize each vertex to be white in color.
    if textures is None:
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)

        textures = Textures(verts_rgb=verts_rgb.to(device))

    faces = torch.tensor(np.int32(faces), dtype=torch.long).cuda()

    # return Meshes(
    #     verts=[verts.to(device)],
    #     faces=[faces.to(device)],
    #     textures=textures
    # )

    return Meshes(
        verts=verts.to(device),
        faces=faces.to(device).repeat(verts.shape[0],1,1),
        textures=textures.to(device)
    )



