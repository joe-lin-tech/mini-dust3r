import rerun as rr
from pathlib import Path
from typing import Literal
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
from smplx import SMPL
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
SCENE_GRAPH = "window-10"
DOWNSAMPLE_FACTOR = 1
USE_GT_POSE = False
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
        pred_log_path = f"{parent_log_path}/pred_camera"
        gt_log_path = f"{parent_log_path}/gt_camera"
        rr.set_time_sequence("frame_idx", i)

        rr.log(
            f"{pred_log_path}",
            rr.Transform3D(
                translation=frame_result["pred_cam"][:3, 3],
                mat3x3=frame_result["pred_cam"][:3, :3],
                from_parent=False,
            ),
        )

        rr.log(
            f"{pred_log_path}/pinhole",
            rr.Pinhole(
                image_from_camera=intrinsics,
                height=optimized_result.img_size[1],
                width=optimized_result.img_size[0],
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )

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
            rr.log(
                f"{pred_log_path}/pinhole/rgb",
                rr.Image(rgb_hw3),
            )
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
                     vis_min_conf_thr: int) -> OptimizedResult:
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
    smpl_file = os.path.join(ROOT_PATH, "tram_new", "hps", "vimo_track_0.npy")

    pred_smpl = np.load(smpl_file, allow_pickle=True).item()
    pred_rotmat = pred_smpl['pred_rotmat']
    pred_shape = pred_smpl['pred_shape']
    pred_trans = pred_smpl['pred_trans']
    pred_frame = pred_smpl['frame'].tolist()

    # use mean shape as body shape
    mean_shape = pred_shape.mean(dim=0, keepdim=True)
    pred_shape = mean_shape.repeat(len(pred_shape), 1)

    smpl = SMPL(model_path="data/smpl")
    J_regressor_feet = torch.from_numpy(np.load("data/smpl/J_regressor_feet.npy")).float()
    tt = lambda x: torch.Tensor(x).float()

    pred_smpl = smpl(body_pose=pred_rotmat[:, 1:],
                     global_orient=pred_rotmat[:, [0]],
                     betas=pred_shape,
                     transl=pred_trans.squeeze(),
                     pose2rot=False,
                     default_smpl=True)
    
    feet_joints_all = vertices2joints(J_regressor_feet, pred_smpl.vertices)

    with open(CONTACT_PKL_PATH, 'rb') as f:
        contact_file = pickle.load(f)
        contact_file = contact_file[0]
        assert contact_file.shape[0] == feet_joints_all.shape[0]

    # determine initial scale factor
    smpl_scale_info = []
    smpl_contact_info = []
    all_scales = []
    w, h = rgb_hw3_list[0].shape[1], rgb_hw3_list[0].shape[0]

    last_contact_joints = []
    last_image_frame = 0
    last_frame = 0

    for i in range(NUM_FRAMES):
        current_idx = i + START_FRAME
        if current_idx in pred_frame and i in IMG_IDX:
            frame = pred_frame.index(current_idx)
            image_frame = IMG_IDX.index(i)
            feet_joints = feet_joints_all[frame].cpu().numpy()
            contact_prob = contact_file[frame]
            
            feet_joints_2d = np.einsum("ij,nj->ni", K_b33[0], feet_joints)
            feet_joints_2d = feet_joints_2d / feet_joints_2d[:, [2]]
            feet_joints_2d = feet_joints_2d[:, :2]

            new_contact_joints = []
            for joint_idx in [0, 2]:  # two toe joints
                # ensure the joints are projected on the image:
                pred_feet_2d = feet_joints_2d[joint_idx]
                if 0 <= pred_feet_2d[0] < w and 0 <= pred_feet_2d[1] < h and contact_prob[joint_idx] > 0.3:
                    pred_feet_3d = feet_joints[joint_idx]
                    valid_feet_2d = find_closest_point(masks_list[image_frame],
                                                        pred_feet_2d[1],
                                                        pred_feet_2d[0])
                    valid_feet_depth = depth_hw_list[image_frame][
                        valid_feet_2d[1], valid_feet_2d[0]]
                    scale_factor = pred_feet_3d[2] / valid_feet_depth
                    smpl_scale_info.append(torch.tensor([image_frame, valid_feet_2d[1], valid_feet_2d[0], pred_feet_3d[2]]).cuda())
                    all_scales.append(scale_factor)
                    new_contact_joints.append(joint_idx)
                    if joint_idx in last_contact_joints and image_frame - last_image_frame == 1: # only allow 5 frame intervals
                        smpl_contact_info.append(torch.concatenate([
                            torch.tensor([image_frame]), 
                            feet_joints_all[frame][joint_idx], 
                            torch.ones(1),
                            torch.tensor([last_image_frame]), 
                            feet_joints_all[last_frame][joint_idx],
                            torch.ones(1)
                        ]).cuda())
            last_contact_joints = new_contact_joints
            last_image_frame = image_frame
            last_frame = frame

    assert len(all_scales) > 0
    all_scales = np.array(all_scales)
    scale_factor = np.median(all_scales)

    # optimize scale factor with smpl info
    print(f"scale before optimization: {scale_factor}")
    new_scene = PointCloudOptimizerSMPL(scene, torch.stack(smpl_scale_info), torch.stack(smpl_contact_info), scale_factor)
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

    pred_cam = interpolate_se3(world_T_cam_b44,
                               times=np.array(IMG_IDX),
                               query_times=np.arange(NUM_FRAMES))
    # add smpl mesh in global
    pred_smpl_mesh = []
    pred_joints_all = []

    SMPL_IDX = []
    for i in range(NUM_FRAMES):
        current_idx = i + START_FRAME
        if current_idx in pred_frame:
            SMPL_IDX.append(i)
            frame = pred_frame.index(current_idx)
            camera_transform = tt(pred_cam[i])
            pred_vertices = pred_smpl.vertices[frame]  # hardcoded interval
            pred_joints = pred_smpl.joints[frame, :24]  # hardcoded interval
            pred_vertices = torch.concatenate(
                [pred_vertices,
                 torch.ones([pred_vertices.shape[0], 1])],
                dim=-1)
            pred_joints = torch.concatenate(
                [pred_joints,
                 torch.ones([pred_joints.shape[0], 1])], axis=-1)
            pred_vertices = torch.einsum('ij,nj->ni', camera_transform,
                                         pred_vertices)
            pred_joints = torch.einsum('ij,nj->ni', camera_transform,
                                       pred_joints)
            pred_smpl_mesh.append(
                trimesh.Trimesh(vertices=pred_vertices[:, :3],
                                faces=smpl.faces))
            pred_joints_all.append(pred_joints[:, :3])

    # add gt smpl paramters to optimized results
    with open(GT_PKL_PATH, 'rb') as f:
        gt_file = pickle.load(f)

    gt_camera = gt_file["camera"]["extrinsics"]  # w to c

    init_trans = gt_camera[START_FRAME]  # w to c0
    gt_camera = np.linalg.inv(gt_camera)  # c to w
    gt_camera = np.einsum("ij,njk->nik", init_trans, gt_camera)  # c to c0

    init_trans = tt(init_trans)

    gt_global_orient = axis_angle_to_matrix(tt(gt_file["smpl"]["poses_root"]))
    gt_transl = tt(gt_file["smpl"]["trans"])

    smpl_gt = SMPL(model_path="data/smpl", gender=gt_file["gender"])
    gt_smpl = smpl_gt(body_pose=axis_angle_to_matrix(
        tt(gt_file["smpl"]["poses_body"]).reshape(-1, 23, 3)),
                      global_orient=gt_global_orient.unsqueeze(1),
                      betas=tt(gt_file["smpl"]["betas"]).unsqueeze(0).repeat(
                          gt_file["smpl"]["poses_body"].shape[0], 1),
                      transl=gt_transl,
                      pose2rot=False,
                      default_smpl=True)

    gt_cam = []
    gt_smpl_mesh = []
    gt_joints_all = []
    for i in range(NUM_FRAMES):
        current_idx = i + START_FRAME
        gt_cam.append(gt_camera[current_idx])

        if gt_file["good_frames_mask"][current_idx]:
            gt_vertices = gt_smpl.vertices[current_idx]
            gt_vertices = torch.concatenate(
                [gt_vertices,
                 torch.ones([gt_vertices.shape[0], 1])], axis=-1)
            gt_vertices = torch.einsum('ij,nj->ni', init_trans, gt_vertices)
            gt_smpl_mesh.append(
                trimesh.Trimesh(vertices=gt_vertices[:, :3], faces=smpl.faces))
            gt_joints = gt_smpl.joints[current_idx, :24]
            gt_joints = torch.concatenate(
                [gt_joints, torch.ones([gt_joints.shape[0], 1])], dim=-1)
            gt_joints = torch.einsum('ij,nj->ni', init_trans, gt_joints)
            gt_joints_all.append(gt_joints[:, :3])

    gt_cam = np.stack(gt_cam, axis=0)

    assert len(gt_joints_all) == len(pred_joints_all)

    target_j3d = torch.stack(gt_joints_all[:EVAL_CHUNK_SIZE], dim=0)
    pred_j3d = torch.stack(pred_joints_all[:EVAL_CHUNK_SIZE], dim=0)

    w_j3d = first_align_joints(target_j3d, pred_j3d)
    wa_j3d = global_align_joints(target_j3d, pred_j3d)

    w_jpe = compute_jpe(target_j3d, w_j3d) * 1000
    wa_jpe = compute_jpe(target_j3d, wa_j3d) * 1000

    print(f"w_jpe: {w_jpe.mean()}")
    print(f"wa_jpe: {wa_jpe.mean()}")

    frame_result = []
    for i in range(NUM_FRAMES):
        per_frame_result = OptimizedFrameResult(frame_idx=i,
                                                gt_cam=gt_cam[i],
                                                pred_cam=pred_cam[i])

        if i in SMPL_IDX:
            idx = SMPL_IDX.index(i)
            per_frame_result["pred_smpl"] = pred_smpl_mesh[idx]
            per_frame_result["gt_smpl"] = gt_smpl_mesh[idx]

        if i in IMG_IDX:
            idx = IMG_IDX.index(i)
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
            "w_jpe": w_jpe,
            "wa_jpe": wa_jpe
        })

    return optimised_result


def inferece_dust3r(
    image_dir_or_list: Path | list[Path] | None,
    model: AsymmetricCroCo3DStereo,
    device: Literal["cpu", "cuda", "mps"],
    batch_size: int = 1,
    image_size: Literal[224, 512] = 512,
    schedule: Literal["linear", "cosine"] = "linear",
    vis_min_conf_thr: float = 10,
) -> OptimizedResult:
    """
    Perform inference using the Dust3r algorithm.

    Args:
        image_dir_or_list (Union[Path, List[Path]]): Path to the directory containing images or a list of image paths.
        model (AsymmetricCroCo3DStereo): The Dust3r model to use for inference.
        device (Literal["cpu", "cuda", "mps"]): The device to use for inference ("cpu", "cuda", or "mps").
        batch_size (int, optional): The batch size for inference. Defaults to 1.
        image_size (Literal[224, 512], optional): The size of the input images. Defaults to 512.
        niter (int, optional): The number of iterations for the global alignment optimization. Defaults to 100.
        schedule (Literal["linear", "cosine"], optional): The learning rate schedule for the global alignment optimization. Defaults to "linear".
        min_conf_thr (float, optional): The minimum confidence threshold for the optimized result. Defaults to 10.

    Returns:
        OptimizedResult: The optimized result containing the RGB, depth, and confidence images.

    Raises:
        ValueError: If `image_dir_or_list` is neither a list of paths nor a path.
    """
    imgs, masks, num_frame, img_idx = load_images(
        folder_or_list=ROOT_PATH + "/images",
        size=image_size,
        verbose=True,
        interval=SAMPLE_INTERVAL,
        max_num=MAX_NUM,
        start_frame=START_FRAME,
    )
    assert len(masks) == len(imgs)

    globals()["NUM_FRAMES"] = num_frame
    globals()["IMG_IDX"] = img_idx

    # if only one image was loaded, duplicate it to feed into stereo network
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1

    pairs: list[tuple[ImageDict,
                      ImageDict]] = make_pairs(imgs,
                                               scene_graph=SCENE_GRAPH,
                                               prefilter=None,
                                               symmetrize=True)

    mask_pairs: list[tuple[ImageDict,
                           ImageDict]] = make_pairs(masks,
                                                    scene_graph=SCENE_GRAPH,
                                                    prefilter=None,
                                                    symmetrize=True)

    output: Dust3rResult = inference(pairs,
                                     model,
                                     device,
                                     batch_size=batch_size)

    for i, (mask1, mask2) in enumerate(mask_pairs):
        output["pred1"]["conf"][i][mask1["img"][0][0] == 0] = 1.0
        output["pred2"]["conf"][i][mask2["img"][0][0] == 0] = 1.0

    # downsample the images
    output["view1"]["img"] = output["view1"][
        "img"][:, :, ::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR].contiguous()
    output["view2"]["img"] = output["view2"][
        "img"][:, :, ::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR].contiguous()
    output["pred1"]["conf"] = output["pred1"][
        "conf"][:, ::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR].contiguous()
    output["pred2"]["conf"] = output["pred2"][
        "conf"][:, ::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR].contiguous()
    output["pred1"]["pts3d"] = output["pred1"][
        "pts3d"][:, ::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR].contiguous()
    output["pred2"]["pts3d_in_other_view"] = output["pred2"][
        "pts3d_in_other_view"][:, ::DOWNSAMPLE_FACTOR, ::
                               DOWNSAMPLE_FACTOR].contiguous()

    output["view1"][
        'true_shape'] = output["view1"]['true_shape'] // DOWNSAMPLE_FACTOR
    output["view2"][
        'true_shape'] = output["view2"]['true_shape'] // DOWNSAMPLE_FACTOR

    mode = (GlobalAlignerMode.PointCloudOptimizer
            if len(imgs) > 2 else GlobalAlignerMode.PairViewer)

    scene: BasePCOptimizer = global_aligner(dust3r_output=output,
                                            device=device,
                                            mode=mode,
                                            optimize_pp=True)

    # read camera gt file
    with open(GT_PKL_PATH, "rb") as f:
        gt_camera = pickle.load(f)
    f = gt_camera["camera"]["intrinsics"][0][0]
    cx, cy = gt_camera["camera"]["intrinsics"][0][2], gt_camera["camera"][
        "intrinsics"][1][2]
    ori_width, ori_height = gt_camera["camera"]["width"], gt_camera["camera"][
        "height"]

    cx = cx * output["view1"]['true_shape'][0][1] / ori_width
    cy = cy * output["view1"]['true_shape'][0][0] / ori_height
    f = f * output["view1"]['true_shape'][0][0] / ori_height

    assert np.allclose(output["view1"]['true_shape'][0][1] / ori_width,
                       output["view1"]['true_shape'][0][0] / ori_height)

    # preset camera intrinsics
    scene.preset_focal(np.array([f]).repeat(len(imgs), axis=0))
    scene.preset_principal_point(
        np.array([[cx, cy]]).repeat(len(imgs), axis=0))

    # preset camera poses
    if USE_GT_POSE:
        poses = gt_camera["camera"]["extrinsics"]
        poses = poses[::SAMPLE_INTERVAL]
        scene.preset_pose(poses)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="known_poses" if USE_GT_POSE else "mst",
            niter=NITER,
            schedule=SCHEDULE,
            lr=LR)

    # get the optimized result from the scene
    optimized_result: OptimizedResult = scene_to_results(
        scene, vis_min_conf_thr)

    # for e in range(len(scene.pw_poses)):
    #     print("after scale optmization:")
    #     print(torch.exp(scene.pw_poses[e].data[-1]))

    # for idx, pose in enumerate(scene.get_pw_poses()):
    #     print(f' (setting pose #{idx} = {pose[:3,3]})')

    return optimized_result

if __name__ == "__main__":
    parser = ArgumentParser("mini-dust3r eval script")
    parser.add_argument(
        "--root_path",
        type=str,
        default=ROOT_PATH
    )

    parser.add_argument(
        "--start_frame",
        type=int,
        default=START_FRAME
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="test"
    )
    args = parser.parse_args()
    ROOT_PATH = args.root_path
    START_FRAME = args.start_frame

    globals()["GT_PKL_PATH"] = glob(f'{ROOT_PATH}/*_data.pkl')[0]
    globals()["CONTACT_PKL_PATH"] = glob(f'{ROOT_PATH}/*contact.pkl')[0]

    with open(GT_PKL_PATH, 'rb') as f:
        gt_file = pickle.load(f)


    # always ensure the first frame is valid
    # while not gt_file["good_frames_mask"][START_FRAME]:
    #     START_FRAME += 1

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = AsymmetricCroCo3DStereo.from_pretrained(
        "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    result = inferece_dust3r(image_dir_or_list=None, model=model,
        device=device,
        batch_size=1,)

    save_result = {}
    save_result["eval_metrics"] = result.eval_metrics
    save_result["pred_cam"] = np.stack([x["pred_cam"] for x in result.frame_result], axis=0)

    sequence_id = ROOT_PATH.split("/")[-1][:2]

    parent_folder = os.path.join("results", args.exp_name)
    os.makedirs(parent_folder, exist_ok=True)

    result_file_name = os.path.join(parent_folder, f"result_{sequence_id}_{START_FRAME}-{START_FRAME+NUM_FRAMES-1}.pkl")

    with open(result_file_name, "wb") as f:
        pickle.dump(save_result, f)
