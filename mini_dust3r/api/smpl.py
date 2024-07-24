import rerun as rr
import rerun.blueprint as rrb
from rerun.components import Material
from pathlib import Path
from argparse import ArgumentParser
import torch
import matplotlib.pyplot as plt
import trimesh

import os
import numpy as np
import pickle
from smplx import SMPLLayer
from mini_dust3r.utils.rot import axis_angle_to_matrix

parser = ArgumentParser("smpl vis script")
rr.script_add_args(parser)
args = parser.parse_args()
rr.script_setup(args, "mini-dust3r")

with open("data/citywalkers/full_all_with_measurements.pkl", "rb") as f:
    all_smpl = pickle.load(f)

image_dir = "2s13b3ctnKs_79"
smpl_motion = [motion for motion in all_smpl if motion['image'].split('/')[0] == image_dir][0]
smpl = SMPLLayer(model_path="data/smpl")
tt = lambda x: torch.Tensor(x).float()

global_orient = smpl_motion['global_orient']
local_orient = smpl_motion['local_orient']
betas = smpl_motion['betas']
global_transl = smpl_motion['global_trans']
local_transl = smpl_motion['local_trans']
body_pose = smpl_motion['body_pose']

global_smpl = smpl(body_pose=axis_angle_to_matrix(tt(body_pose[::20])),
                global_orient=axis_angle_to_matrix(tt(global_orient[::20])),
                betas=tt(betas[::20]),
                transl=tt(global_transl[::20]),
                pose2rot=False,
                default_smpl=True)

for i, vertices in enumerate(global_smpl.vertices):
    mesh = trimesh.Trimesh(vertices=vertices[:, :3], faces=smpl.faces)
    rr.log(
        f"world/global_smpl_{i}",
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            triangle_indices=mesh.faces,
            vertex_normals=mesh.vertex_normals,
            mesh_material=Material(albedo_factor=(0.0, 0.0, 1.0, 1.0)),
        )
    )

# local_smpl = smpl(body_pose=axis_angle_to_matrix(tt(body_pose[::20])),
#                 global_orient=axis_angle_to_matrix(tt(local_orient[::20])),
#                 betas=tt(betas[::20]),
#                 transl=tt(local_orient[::20]),
#                 pose2rot=False,
#                 default_smpl=True)

# for i, vertices in enumerate(local_smpl.vertices):
#     mesh = trimesh.Trimesh(vertices=vertices[:, :3], faces=smpl.faces)
#     rr.log(
#         f"world/local_smpl_{i}",
#         rr.Mesh3D(
#             vertex_positions=mesh.vertices,
#             triangle_indices=mesh.faces,
#             vertex_normals=mesh.vertex_normals,
#             mesh_material=Material(albedo_factor=(0.0, 0.0, 1.0, 1.0)),
#         )
#     )

frame_id = int(smpl_motion['image'].split('/')[-1][:-4]) - 1
global_orient_source = axis_angle_to_matrix(tt(global_orient[frame_id]))
global_orient_target = axis_angle_to_matrix(tt(local_orient[frame_id]))
source_to_target_rotation = global_orient_target @ global_orient_source.T
# rotation matrix from standard frame to initial camera frame

global_orient = axis_angle_to_matrix(tt(global_orient))
global_orient = source_to_target_rotation @ global_orient

transl_source = tt(global_transl[[frame_id]])
transl_target = tt(local_transl[[frame_id]])
source_to_target_translation = transl_target.T - source_to_target_rotation @ transl_source.T
# translation to position in initial camera frame

transl = tt(global_transl)
transl = source_to_target_rotation @ transl.T + source_to_target_translation
transl = transl.T

body_pose = axis_angle_to_matrix(tt(body_pose))
betas = tt(betas)

initial_smpl = smpl(body_pose=tt(body_pose),
            global_orient=tt(global_orient),
            betas=tt(betas),
            transl=tt(transl),
            pose2rot=False,
            default_smpl=True)

for i, vertices in enumerate(initial_smpl.vertices):
    mesh = trimesh.Trimesh(vertices=vertices[:, :3], faces=smpl.faces)
    rr.log(
        f"world/initial_smpl_{i}",
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            triangle_indices=mesh.faces,
            vertex_normals=mesh.vertex_normals,
            mesh_material=Material(albedo_factor=(0.0, 0.0, 1.0, 1.0)),
        )
    )

x_axis = rr.Arrows3D(origins=[0, 0, 0], vectors=[1, 0, 0])
y_axis = rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 1, 0])
z_axis = rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 0, 1])

rr.log("world/x", x_axis)
rr.log("world/y", y_axis)
rr.log("world/z", z_axis)

rr.log("world/global", rr.Points3D(positions=global_transl))
rr.log("world/local", rr.Points3D(positions=local_transl))
rr.script_teardown(args)