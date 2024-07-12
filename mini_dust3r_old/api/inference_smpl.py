import rerun as rr
from pathlib import Path
from typing import Literal, List
import copy
import torch
import numpy as np
import os
from jaxtyping import Float32, Bool
from argparse import ArgumentParser
import pickle
import trimesh
from glob import glob
from tqdm import tqdm

from mini_dust3r_old.utils.image import load_images, ImageDict
from mini_dust3r_old.inference import inference, Dust3rResult
from mini_dust3r_old.model import AsymmetricCroCo3DStereo
from mini_dust3r_old.image_pairs import make_pairs
from mini_dust3r_old.cloud_opt import global_aligner, GlobalAlignerMode
from mini_dust3r_old.cloud_opt.base_opt import BasePCOptimizer
from mini_dust3r_old.cloud_opt.optimizer_smpl import PointCloudOptimizerSMPL
from mini_dust3r_old.cloud_opt.base_opt import global_alignment_loop
from mini_dust3r_old.viz import pts3d_to_trimesh, cat_meshes
from dataclasses import dataclass
from smplx import SMPLLayer
from smplx.lbs import vertices2joints
from rerun.components import Material
from mini_dust3r_old.utils.rot import axis_angle_to_matrix, interpolate_se3
from mini_dust3r_old.utils.eval_utils import first_align_joints, global_align_joints, compute_jpe

from typing import  TypedDict

# sample config
START_FRAME = 1900
SAMPLE_INTERVAL = 5
MAX_NUM = 21

# dust3r config
NITER = 100
SCHEDULE= "linear"
LR = 0.01

# other config
EVAL_CHUNK_SIZE=100
NUM_FRAMES = 1 + (MAX_NUM-1) * SAMPLE_INTERVAL
PRED_SMPL_COLOR = (1.0, 0.0, 0.0, 1.0)
GT_SMPL_COLOR = (0.0, 0.0, 1.0, 1.0)
# ROOT_PATH = "data/EMDB2/09_outdoor_walk"
# GT_PKL_PATH = glob(f'{ROOT_PATH}/*_data.pkl')[0]
# CONTACT_PKL_PATH = glob(f'{ROOT_PATH}/*contact.pkl')[0]
# IMG_IDX=[0] # placeholder


class OptimizedFrameResult(TypedDict):
    frame_idx: int
    conf_hw: Float32[torch.Tensor, "h w"] | None
    rgb_hw3: Float32[np.ndarray, "h w 3"] | None
    depth_hw: Float32[np.ndarray, "h w"] | None
    mask_hw: Bool[np.ndarray, "h w"] | None
    pred_smpl: trimesh.Trimesh | None
    gt_smpl: trimesh.Trimesh | None
    pred_cam: Float32[np.ndarray, "4 4"]
    gt_cam: Float32[np.ndarray, "b 4 4"]



@dataclass
class OptimizedResult:
    point_cloud: trimesh.PointCloud
    mesh: trimesh.Trimesh
    intrinsics: Float32[np.ndarray, "3 3"]
    img_size: tuple[int, int]
    frame_result: list[OptimizedFrameResult]
    eval_metrics: dict




def log_optimized_result(
    optimized_result: OptimizedResult, parent_log_path: Path
) -> None:
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
    # log pointcloud
    rr.log(
        f"{parent_log_path}/pointcloud",
        rr.Points3D(
            positions=optimized_result.point_cloud.vertices,
            colors=optimized_result.point_cloud.colors,
        ),
        timeless=True,
    )

    mesh = optimized_result.mesh
    rr.log(
        f"{parent_log_path}/mesh",
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            vertex_colors=mesh.visual.vertex_colors,
            triangle_indices=mesh.faces,
        ),
        timeless=True,
    )

    intrinsics = optimized_result.intrinsics


    pbar = tqdm(
        optimized_result.frame_result,
        total=len(optimized_result.frame_result),
    )
    for i, frame_result in enumerate(pbar):
        # pred_log_path = f"{parent_log_path}/pred_camera"
        gt_log_path = f"{parent_log_path}/gt_camera"
        rr.set_time_sequence("frame_idx", i)

        # rr.log(
        #     f"{pred_log_path}",
        #     rr.Transform3D(
        #         translation=frame_result["pred_cam"][:3, 3],
        #         mat3x3=frame_result["pred_cam"][:3, :3],
        #         from_parent=False,
        #     ),
        # )

        # rr.log(
        #     f"{pred_log_path}/pinhole",
        #     rr.Pinhole(
        #         image_from_camera=intrinsics,
        #         height=optimized_result.img_size[1],
        #         width=optimized_result.img_size[0],
        #         camera_xyz=rr.ViewCoordinates.RDF,
        #     ),
        # )

        rr.log(
            f"{gt_log_path}",
            rr.Transform3D(
                translation=frame_result["gt_cam"][:3, 3],
                mat3x3=frame_result["gt_cam"][:3, :3],
                from_parent=False,
            ),
        )


        rr.log(
            f"{gt_log_path}/pinhole",
            rr.Pinhole(
                image_from_camera=intrinsics,
                height=optimized_result.img_size[1],
                width=optimized_result.img_size[0],
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )

        if "pred_smpl" in frame_result:

            pred_smpl_mesh = frame_result["pred_smpl"]

            rr.log(
                f"{parent_log_path}/pred_smpl_mesh",
                rr.Mesh3D(
                    vertex_positions=pred_smpl_mesh.vertices,
                    triangle_indices=pred_smpl_mesh.faces,
                    vertex_normals=pred_smpl_mesh.vertex_normals,
                    mesh_material=Material(albedo_factor=PRED_SMPL_COLOR),
                ),
            )

            gt_smpl_mesh = frame_result["gt_smpl"]
            rr.log(
                    f"{parent_log_path}/gt_smpl_mesh",
                    rr.Mesh3D(
                        vertex_positions=gt_smpl_mesh.vertices,
                        triangle_indices=gt_smpl_mesh.faces,
                        vertex_normals=gt_smpl_mesh.vertex_normals,
                        mesh_material=Material(albedo_factor=GT_SMPL_COLOR),
                    ),
                )

        if "rgb_hw3" in frame_result:

            rgb_hw3 = frame_result["rgb_hw3"]
            # rr.log(
            #     f"{pred_log_path}/pinhole/rgb",
            #     rr.Image(rgb_hw3),
            # )
            rr.log(
                f"{gt_log_path}/pinhole/rgb",
                rr.Image(rgb_hw3),
            )



def find_closest_point(M, x, y):
    # Get the indices of points in M that have value 1
    points = np.argwhere(M == 1)

    # Compute the distances from (x,y) to each of these points
    distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)

    # Find the index of the minimum distance
    min_index = np.argmin(distances)

    # Get the closest point
    closest_point = points[min_index]

    return  [closest_point[1], closest_point[0]]


def scene_to_results(scene: BasePCOptimizer,
                     vis_min_conf_thr: int,
                     smpl_path: str = "data/citywalkers/full_all_with_measurements.pkl",
                     smpl_model_path: str = "data/smpl",
                     image_dir: str = "data/citywalkers/_fuCbKaSuJ8_77/images",
                     img_indices: List[int] = [0]) -> OptimizedResult:
    # get camera parameters K and T
    K_b33: Float32[np.ndarray,
                   "b 3 3"] = scene.get_intrinsics().numpy(force=True)
    # image, depths, confidence
    rgb_hw3_list: list[Float32[np.ndarray, "h w 3"]] = scene.imgs
    depth_hw_list: list[Float32[np.ndarray, "h w"]] = [
        depth.numpy(force=True) for depth in scene.get_depthmaps()
    ]
    conf_hw_list: list[Float32[np.ndarray, "h w"]] = [
        c.numpy(force=True) for c in scene.get_conf()
    ]
 
    # get log confidence
    log_conf_trf: Float32[torch.Tensor,
                          ""] = scene.conf_trf(torch.tensor(vis_min_conf_thr))
    # set the minimum confidence threshold
    scene.min_conf_thr = float(log_conf_trf)
    # obtain masks based on threshold
    masks_list: Bool[np.ndarray, "h w"] = [
        mask.numpy(force=True) for mask in scene.get_masks()
    ]

    # add smpl paramters to optimized results
    with open(smpl_path, "rb") as f:
        all_smpl = pickle.load(f)

    gt_smpl = [motion for motion in all_smpl if motion['image'].split('/')[0] == image_dir.split('/')[2]][0]

    smpl = SMPLLayer(model_path=smpl_model_path)
    J_regressor_feet = torch.from_numpy(np.load(f"{smpl_model_path}/J_regressor_feet.npy")).float()
    tt = lambda x: torch.Tensor(x).float()

    global_orient = axis_angle_to_matrix(tt(gt_smpl['global_orient']))
    betas = gt_smpl['betas']
    transl = gt_smpl['global_trans']
    body_pose = gt_smpl['body_pose']
    frame_id = int(gt_smpl['image'].split('/')[1][:-4])

    global_orient_source = axis_angle_to_matrix(
        tt(gt_smpl['global_orient'][frame_id]))
    global_orient_target = axis_angle_to_matrix(
        tt(gt_smpl['local_orient'][frame_id]))
    source_to_target_rotation = global_orient_target @ global_orient_source.T

    global_orient = axis_angle_to_matrix(
        tt(gt_smpl['global_orient']))
    global_orient = source_to_target_rotation @ global_orient

    transl_source = tt(gt_smpl['global_trans'][[frame_id]])
    transl_target = tt(gt_smpl['local_trans'][[frame_id]])
    source_to_target_translation = transl_target.T - source_to_target_rotation @ transl_source.T
    
    transl = tt(gt_smpl['global_trans'])
    transl = source_to_target_rotation @ transl.T + source_to_target_translation
    transl = transl.T

    # use mean shape as body shape
    mean_beta = betas.mean(axis=0, keepdims=True)
    betas = mean_beta.repeat(len(betas), 0)

    gt_smpl = smpl(body_pose=axis_angle_to_matrix(tt(body_pose)),
                     global_orient=tt(global_orient),
                     betas=tt(betas),
                     transl=tt(transl),
                     pose2rot=False,
                     default_smpl=True)
    
    feet_joints_all = vertices2joints(J_regressor_feet, gt_smpl.vertices)

    # determine initial scale factor
    smpl_scale_info = []
    all_scales = []
    w, h = rgb_hw3_list[0].shape[1], rgb_hw3_list[0].shape[0]

    for f, frame_path in enumerate(os.listdir(image_dir)):
        i = int(frame_path[:-4]) - 1
        if i in img_indices:
            feet_joints = feet_joints_all[i].cpu().numpy()
            feet_joints_2d = np.einsum("ij,nj->ni", K_b33[0], feet_joints)
            feet_joints_2d = feet_joints_2d / feet_joints_2d[:, [2]]
            feet_joints_2d = feet_joints_2d[:, :2]

            new_contact_joints = []
            for joint_idx in [0, 2]:  # two toe joints
                # ensure the joints are projected on the image:
                gt_feet_2d = feet_joints_2d[joint_idx]
                if 0 <= gt_feet_2d[0] < w and 0 <= gt_feet_2d[1] < h:
                    gt_feet_3d = feet_joints[joint_idx]
                    valid_feet_2d = find_closest_point(masks_list[i],
                                                        gt_feet_2d[1],
                                                        gt_feet_2d[0])
                    valid_feet_depth = depth_hw_list[i][valid_feet_2d[1], valid_feet_2d[0]]
                    scale_factor = gt_feet_3d[2] / valid_feet_depth
                    smpl_scale_info.append(torch.tensor([f, valid_feet_2d[1], valid_feet_2d[0], gt_feet_3d[2]]).cuda())
                    all_scales.append(scale_factor)
                    new_contact_joints.append(joint_idx)


    assert len(all_scales) > 0
    all_scales = np.array(all_scales)
    scale_factor = np.median(all_scales)

    # optimize scale factor with smpl info
    print(f"scale before optimization: {scale_factor}")
    new_scene = PointCloudOptimizerSMPL(scene, torch.stack(smpl_scale_info), scale_factor)
    with torch.autograd.set_detect_anomaly(True):
        global_alignment_loop(new_scene,
                niter=NITER,
                schedule=SCHEDULE,
                lr=LR)
    scale_factor = new_scene.global_scale.exp().item()
    print(f"scale after optimization: {scale_factor}")
    
    # update pose, depth and pts
    world_T_cam_b44: Float32[np.ndarray,
                             "b 4 4"] = new_scene.get_im_poses().numpy(force=True)
    depth_hw_list: list[Float32[np.ndarray, "h w"]] = [
        depth.numpy(force=True) for depth in new_scene.get_depthmaps()
    ]

    # point cloud, mesh
    pts3d_list: list[Float32[np.ndarray, "h w 3"]] = [
        pt3d.numpy(force=True) for pt3d in new_scene.get_pts3d(global_coord=False)
    ]

    # normalize the point cloud and apply the scale
    normalize_transform = np.linalg.inv(world_T_cam_b44[0])

    world_T_cam_b44 = np.einsum('ij,bjk->bik', normalize_transform,
                                world_T_cam_b44)
    world_T_cam_b44[:, :3, 3] = world_T_cam_b44[:, :3, 3] * scale_factor
    for i, pts3d in enumerate(pts3d_list):
        pts3d = pts3d * scale_factor
        pts3d = np.concatenate(
            [pts3d, np.ones([pts3d.shape[0], pts3d.shape[1], 1])], axis=-1)
        pts3d = np.einsum('ij,hwj->hwi', world_T_cam_b44[i], pts3d)
        pts3d_list[i] = pts3d[..., :3]

    for i, depth in enumerate(depth_hw_list):
        depth = depth * scale_factor
        depth_hw_list[i] = depth

    point_cloud: Float32[np.ndarray, "num_points 3"] = np.concatenate(
        [p[m] for p, m in zip(pts3d_list, masks_list)])
    colors: Float32[np.ndarray, "num_points 3"] = np.concatenate(
        [p[m] for p, m in zip(rgb_hw3_list, masks_list)])
    point_cloud = trimesh.PointCloud(point_cloud.reshape(-1, 3),
                                     colors=colors.reshape(-1, 3))

    meshes = []
    pbar = tqdm(zip(rgb_hw3_list, pts3d_list, masks_list),
                total=len(rgb_hw3_list))
    for rgb_hw3, pts3d, mask in pbar:
        meshes.append(pts3d_to_trimesh(rgb_hw3, pts3d, mask))

    mesh = trimesh.Trimesh(**cat_meshes(meshes))

    # gt_cam = interpolate_se3(world_T_cam_b44,
    #                            times=np.array(img_indices),
    #                            query_times=np.arange(len(os.listdir(image_dir))))
    # add smpl mesh in global
    gt_smpl_mesh = []
    gt_joints_all = []

    SMPL_IDX = []
    for i in range(len(os.listdir(image_dir))):
        # i = int(frame_path[:-4]) - 1
        SMPL_IDX.append(i)
        camera_transform = tt(world_T_cam_b44[i]) # tt(gt_cam[i])
        gt_vertices = gt_smpl.vertices[i]  # hardcoded interval
        gt_joints = gt_smpl.joints[i, :24]  # hardcoded interval
        gt_vertices = torch.concatenate(
            [gt_vertices,
                torch.ones([gt_vertices.shape[0], 1])],
            dim=-1)
        gt_joints = torch.concatenate(
            [gt_joints,
                torch.ones([gt_joints.shape[0], 1])], axis=-1)
        gt_vertices = torch.einsum('ij,nj->ni', camera_transform,
                                        gt_vertices)
        gt_joints = torch.einsum('ij,nj->ni', camera_transform,
                                    gt_joints)
        gt_smpl_mesh.append(
            trimesh.Trimesh(vertices=gt_vertices[:, :3],
                            faces=smpl.faces))
        gt_joints_all.append(gt_joints[:, :3])

    frame_result = []
    for i in range(len(os.listdir(image_dir))):
        per_frame_result = OptimizedFrameResult(frame_idx=i,
                                                gt_cam=world_T_cam_b44[i])
                                                # gt_cam=gt_cam[i])

        if i in SMPL_IDX:
            idx = SMPL_IDX.index(i)
            per_frame_result["gt_smpl"] = gt_smpl_mesh[idx]

        if i in img_indices:
            idx = img_indices.index(i)
            per_frame_result["rgb_hw3"] = rgb_hw3_list[idx]
            per_frame_result["depth_hw"] = depth_hw_list[idx]
            per_frame_result["conf_hw"] = conf_hw_list[idx]
            per_frame_result["mask_hw"] = masks_list[idx]
        frame_result.append(per_frame_result)

    optimised_result = OptimizedResult(
        point_cloud=point_cloud,
        mesh=mesh,
        intrinsics=K_b33[0],  # intrinsics is fixed
        img_size=(rgb_hw3_list[0].shape[1], rgb_hw3_list[0].shape[0]),
        frame_result=frame_result,
        eval_metrics={
            "w_jpe": 0,
            "wa_jpe": 0
        })

    return optimised_result


def inference_dust3r(
    image_dir: str | None,
    model: AsymmetricCroCo3DStereo,
    device: Literal["cpu", "cuda", "mps"],
    batch_size: int = 1,
    image_size: Literal[224, 512] = 512,
    vis_min_conf_thr: float = 10,
    scene_graph: str = "window-10"
) -> OptimizedResult:
    """
    Perform inference using the Dust3r algorithm.

    Args:
        image_dir (str): Path to the directory containing images.
        model (AsymmetricCroCo3DStereo): The Dust3r model to use for inference.
        device (Literal["cpu", "cuda", "mps"]): The device to use for inference ("cpu", "cuda", or "mps").
        batch_size (int, optional): The batch size for inference. Defaults to 1.
        image_size (Literal[224, 512], optional): The size of the input images. Defaults to 512.
        niter (int, optional): The number of iterations for the global alignment optimization. Defaults to 100.
        min_conf_thr (float, optional): The minimum confidence threshold for the optimized result. Defaults to 10.

    Returns:
        OptimizedResult: The optimized result containing the RGB, depth, and confidence images.

    Raises:
        ValueError: If `image_dir` is not a path.
    """
    imgs, masks, num_frame, img_idx = load_images(
        folder_or_list=image_dir,
        size=image_size,
        verbose=True,
        interval=1,
        max_num=MAX_NUM,
        start_frame=0,
    )
    assert len(masks) == len(imgs)

    # if only one image was loaded, duplicate it to feed into stereo network
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1

    pairs: list[tuple[ImageDict,
                      ImageDict]] = make_pairs(imgs,
                                               scene_graph=scene_graph,
                                               prefilter=None,
                                               symmetrize=True)

    mask_pairs: list[tuple[ImageDict,
                           ImageDict]] = make_pairs(masks,
                                                    scene_graph=scene_graph,
                                                    prefilter=None,
                                                    symmetrize=True)

    output: Dust3rResult = inference(pairs,
                                     model,
                                     device,
                                     batch_size=batch_size)

    for i, (mask1, mask2) in enumerate(mask_pairs):
        output["pred1"]["conf"][i][mask1["img"][0][0] == 0] = 1.0
        output["pred2"]["conf"][i][mask2["img"][0][0] == 0] = 1.0

    mode = (GlobalAlignerMode.PointCloudOptimizer
            if len(imgs) > 2 else GlobalAlignerMode.PairViewer)

    scene: BasePCOptimizer = global_aligner(dust3r_output=output,
                                            device=device,
                                            mode=mode,
                                            optimize_pp=True)

    # use camera parameter assumptions
    # img_w = 1280
    # img_h = 720
    # f = (img_h ** 2 + img_w ** 2) ** 0.5
    # cx = 0.5 * img_w
    # cy = 0.5 * img_h

    # cx = cx * output["view1"]['true_shape'][0][1] / img_w
    # cy = cy * output["view1"]['true_shape'][0][0] / img_h
    # f = f * output["view1"]['true_shape'][0][0] / img_h

    # assert np.allclose(output["view1"]['true_shape'][0][1] / img_w,
    #                    output["view1"]['true_shape'][0][0] / img_h)

    # # preset camera intrinsics
    # scene.preset_focal(np.array([f]).repeat(len(imgs), axis=0))
    # scene.preset_principal_point(
    #     np.array([[cx, cy]]).repeat(len(imgs), axis=0))

    # get the optimized result from the scene
    optimized_result: OptimizedResult = scene_to_results(
        scene, vis_min_conf_thr, image_dir=image_dir)

    return optimized_result