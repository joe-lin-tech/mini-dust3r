import pickle
import os
import numpy as np
import torch
from smplx import SMPL
import cv2
from mini_dust3r_old.utils.rot import axis_angle_to_matrix



image = cv2.imread("data/EMDB2/09_outdoor_walk/images/00000.jpg")

with open(
        os.path.join("data/EMDB2/09_outdoor_walk",
                        "P0_09_outdoor_walk_data.pkl"), 'rb') as f:
    gt_file = pickle.load(f)

gt_camera = gt_file["camera"]["extrinsics"]  # assume w to c
intrinsics =   gt_file["camera"]["intrinsics"]

smpl = SMPL(model_path="data/smpl", gender=gt_file["gender"])

# yup2ydown = axis_angle_to_matrix(torch.tensor([np.pi, 0, 0])).float()

# yup2ydown_numpy = np.eye(4)
# yup2ydown_numpy[:3, :3] = yup2ydown.numpy()
# gt_camera = np.einsum("ij,njk->nik", yup2ydown_numpy, gt_camera)

init_trans = gt_camera[0]  # w to c0
gt_camera = np.linalg.inv(gt_camera) # c to w
gt_camera = np.einsum("ij,njk->nik", init_trans, gt_camera) # c to c0

tt = lambda x: torch.Tensor(x).float()
init_trans = tt(init_trans)

gt_global_orient = axis_angle_to_matrix(tt(gt_file["smpl"]["poses_root"]))
gt_transl = tt(gt_file["smpl"]["trans"])
# gt_global_orient = torch.einsum('ij,bjk->bik', yup2ydown, gt_global_orient)
# gt_transl = torch.einsum('ij,bj->bi', yup2ydown, gt_transl)


# gt_global_orient = torch.einsum('ij,bjk->bik', init_trans[:3, :3],
#                                 gt_global_orient)
# gt_transl = torch.cat(
#     [gt_transl, torch.ones(gt_transl.shape[0], 1)], dim=-1)
# gt_transl = torch.einsum('ij,bj->bi', init_trans, gt_transl)
# gt_transl = gt_transl[:, :3]
gt_smpl = smpl(body_pose=axis_angle_to_matrix(
    tt(gt_file["smpl"]["poses_body"]).reshape(-1, 23, 3)),
               global_orient=gt_global_orient.unsqueeze(1),
               betas=tt(gt_file["smpl"]["betas"]).unsqueeze(0).repeat(
                   gt_file["smpl"]["poses_body"].shape[0], 1),
               transl=gt_transl,
               pose2rot=False,
               default_smpl=True)

gt_joints = gt_smpl.joints[0, :24]

gt_joints = np.concatenate(
            [gt_joints,
             np.ones([gt_joints.shape[0], 1])], axis=-1)
gt_joints = np.einsum('ij,nj->ni', init_trans, gt_joints)

gt_joints = gt_joints[:, :3]

for j in gt_joints:
    x = j[0] / j[2] * intrinsics[0, 0] + intrinsics[0, 2]
    y = j[1] / j[2] * intrinsics[0, 0] + intrinsics[1, 2]
    cv2.circle(image, [int(x), int(y)], 3, [255,0,0], -1)



cv2.imwrite("test_emdb_label.jpg", image)
