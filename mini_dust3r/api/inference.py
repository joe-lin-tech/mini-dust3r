import rerun as rr
from rerun.components import Material
from rerun.datatypes import Rotation3D
from pathlib import Path
from typing import Literal, List
import copy
import torch
import numpy as np
import pickle
from jaxtyping import Float32, Bool
import trimesh
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import point_cloud_utils as pcu
from scipy.interpolate import griddata
import json

from mini_dust3r.utils.image import load_images, ImageDict, MaskDict
from mini_dust3r.inference import inference, Dust3rResult
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.image_pairs import make_pairs
from mini_dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from mini_dust3r.cloud_opt.base_opt import BasePCOptimizer, global_alignment_loop
from mini_dust3r.cloud_opt.optimizer import ScaleOptimizer
from mini_dust3r.viz import pts3d_to_trimesh, cat_meshes
from mini_dust3r.utils.rot import axis_angle_to_matrix, interpolate_se3
from dataclasses import dataclass
from smplx import SMPLLayer
from smplx.lbs import vertices2joints


@dataclass
class OptimizedResult:
    K_b33: Float32[np.ndarray, "b 3 3"]
    world_T_cam_b44: Float32[np.ndarray, "b 4 4"]
    rgb_hw3_list: list[Float32[np.ndarray, "h w 3"]]
    depth_hw_list: list[Float32[np.ndarray, "h w"]]
    conf_hw_list: list[Float32[np.ndarray, "h w"]]
    masks_list: Bool[np.ndarray, "h w"]
    point_cloud: trimesh.PointCloud
    initial_point_cloud: trimesh.PointCloud
    mesh: trimesh.Trimesh
    initial_mesh: trimesh.Trimesh
    gt_smpl_mesh: trimesh.Trimesh
    transl: Float32[np.ndarray, "b 3"]


def log_optimized_result(
    optimized_result: OptimizedResult, parent_log_path: Path
) -> None:
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, timeless=True)
    # log pointcloud
    rr.log(
        f"{parent_log_path}/pointcloud",
        rr.Points3D(
            positions=optimized_result.point_cloud.vertices,
            colors=optimized_result.point_cloud.colors,
        ),
    )

    rr.log(
        f"{parent_log_path}/initial",
        rr.Points3D(
            positions=optimized_result.initial_point_cloud.vertices,
            colors=optimized_result.initial_point_cloud.colors
        )
    )

    # mesh = optimized_result.mesh
    # rr.log(
    #     f"{parent_log_path}/mesh",
    #     rr.Mesh3D(
    #         vertex_positions=mesh.vertices,
    #         vertex_colors=mesh.visual.vertex_colors,
    #         triangle_indices=mesh.faces,
    #     ),
    #     # timeless=True,
    # )

    pbar = tqdm(
        zip(
            optimized_result.rgb_hw3_list,
            optimized_result.depth_hw_list,
            optimized_result.K_b33,
            optimized_result.world_T_cam_b44
        ),
        total=len(optimized_result.rgb_hw3_list),
    )
    for i, (rgb_hw3, depth_hw, k_33, world_T_cam_44) in enumerate(pbar):
        camera_log_path = f"{parent_log_path}/camera_{i}"
        height, width, _ = rgb_hw3.shape
        rr.log(
            f"{camera_log_path}",
            rr.Transform3D(
                translation=world_T_cam_44[:3, 3],
                mat3x3=world_T_cam_44[:3, :3],
                from_parent=False,
            ),
        )
        rr.log(
            f"{camera_log_path}/pinhole",
            rr.Pinhole(
                image_from_camera=k_33,
                height=height,
                width=width,
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )
        rr.log(
            f"{camera_log_path}/pinhole/rgb",
            rr.Image(rgb_hw3),
        )
        # rr.log(
        #     f"{camera_log_path}/pinhole/depth",
        #     rr.DepthImage(depth_hw),
        #     timeless=True
        # )

    for i, gt_smpl_mesh in enumerate(optimized_result.gt_smpl_mesh):
        # display smpl meshes
        rr.log(
            f"{parent_log_path}/gt_smpl_mesh_{i}",
            rr.Mesh3D(
                vertex_positions=gt_smpl_mesh.vertices,
                triangle_indices=gt_smpl_mesh.faces,
                vertex_normals=gt_smpl_mesh.vertex_normals,
                mesh_material=Material(albedo_factor=(0.0, 0.0, 1.0, 1.0)),
            ),
            timeless=True
        )

    rr.log(
        f"{parent_log_path}/boxes",
        rr.Transform3D(
            translation=optimized_result.world_T_cam_b44[0, :3, 3],
            mat3x3=optimized_result.world_T_cam_b44[0, :3, :3],
            from_parent=False,
        ),
    )
    boxes_dict = np.load("debug/boxes.npy", allow_pickle=True)
    boxes_dict = boxes_dict.item()
    boxes = boxes_dict["boxes"]
    classes = boxes_dict["phrases"]
    depth_by_class = { # different depths at different angles
        "car": 2,
        "person": 0.35,
        "fence": 0.1,
        "tree": 0.3,
        "bicycle": 0.1,
        "sign": 0.05,
        "trashcan": 0.25,
        "post": 0.02,
        "bus": 12,
        "phonebooth": 1.5
    }

    lr_thres = 1
    for i, box in enumerate(boxes):
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        cl = int(box[0] + (cx - box[0]) / 2)
        cr = int(box[2] - (box[2] - cx) / 2)
        box = np.vstack([box.reshape(2, 2).T, np.ones((1, 2))])
        depth = boxes_dict["depth"] if classes[i] == "person" else optimized_result.depth_hw_list[0] # generalize to moving objects
        # depth = boxes_dict["depth"]

        dl, dr = depth[cy, cl], depth[cy, cr]

        mid = np.argmin(depth[cy, cl:cr])
        dm = depth[cy, mid + cl]
        mid = np.array([[box[0, 0] + mid, (box[1, 0] + box[1, 1]) / 2, 1]]).T * dm
        mid = np.linalg.inv(optimized_result.K_b33[0]) @ mid
        box *= depth[cy, cx] # [dl, dr] if dl - dr > lr_thres else depth[cy, cx]

        box = np.linalg.inv(optimized_result.K_b33[0]) @ box
        
        if False: # dl - dr > lr_thres: # assume rotated
            obj_w = np.sqrt((mid[0, 0] - box[0, 0]) ** 2 + (mid[2, 0] - box[2, 0]) ** 2)
            obj_h = np.abs(box[1, 1] - box[1, 0])
            obj_d = np.sqrt((box[0, 1] - mid[0, 0]) ** 2 + (box[2, 1] - mid[2, 0]) ** 2)
            obj_cx = (min(box[0]) + max(box[0])) / 2
            obj_cy = (min(box[1]) + max(box[1])) / 2
            obj_cz = (box[2, 0] + box[2, 1]) / 2
        else: # assume front plane parallel
            obj_w = np.abs(box[0, 1] - box[0, 0])
            obj_h = np.abs(box[1, 1] - box[1, 0])
            obj_d = depth_by_class[classes[i]] if classes[i] in depth_by_class else 3
            obj_cx = min(box[0]) + obj_w / 2
            obj_cy = min(box[1]) + obj_h / 2
            obj_cz = min(box[2]) + obj_d / 2

        rr.log(
            f"{parent_log_path}/boxes/box_{i}",
            rr.Boxes3D(
                sizes=[obj_w, obj_h, obj_d],
                centers=[obj_cx, obj_cy, obj_cz],
                labels=classes[i]
            )
        )

    vertices = np.array(optimized_result.initial_mesh.vertices)
    normals = np.array(optimized_result.initial_mesh.vertex_normals)
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    mask = (np.argmax(np.abs(normals), axis=-1) == 1) & (normals[:, 1] > 0) & \
        (vertices[:, 1] >= optimized_result.gt_smpl_mesh[0].vertices[:, 1].min())
    vertices = vertices[mask]
    normals = normals[mask]

    # arrow_log_path = f"{parent_log_path}/arrow_{i}"
    # rr.log(
    #     f"{arrow_log_path}",
    #     rr.Arrows3D(
    #         origins=vertices,
    #         vectors=normals
    #     )
    # )

    points = vertices

    grid_size = [-4, 4, -4, 4]
    grid_points = [40, 40]
    ground_map = np.zeros((grid_points[0], grid_points[1]))

    size_x = (grid_size[1] - grid_size[0]) / grid_points[0]
    size_z = (grid_size[3] - grid_size[2]) / grid_points[1]
    size = np.array([size_x, size_z])
    lower = np.array([grid_size[0], grid_size[2]])

    points[:, [0, 2]] -= optimized_result.transl[[0, 2]]
    points = points[(grid_size[0] <= points[:, 0]) & (points[:, 0] <= grid_size[1])
                    & (grid_size[2] <= points[:, 2]) & (points[:, 2] <= grid_size[3])]
    positions = np.copy(points)
    positions[:, [0, 2]] += optimized_result.transl[[0, 2]]

    rr.log(
        f"{parent_log_path}/ground",
        rr.Points3D(
            positions=positions,
            colors=[0, 255, 0]
        ),
        timeless=True,
    )
    indices = (points[:, [0, 2]] - lower) // size
    indices = indices.astype(np.int32)
    ground_map[indices[:, 0], indices[:, 1]] = points[:, 1]

    try:
        interpolated = griddata(np.argwhere(ground_map != 0.), ground_map[ground_map != 0.],
            np.argwhere(ground_map == 0.), method='linear', fill_value=0.)
        ground_map[ground_map == 0.] = interpolated
    except:
        pass

    return ground_map
    

def find_closest_point(mask, x, y):
    points = np.argwhere(mask == 1)
    dist = np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2)
    min_index = np.argmin(dist)
    closest_point = points[min_index]
    return [closest_point[1], closest_point[0]]

def get_valid_indices(indices, shape):
    indices[:, 0] = np.clip(indices[:, 0], 0, shape[1] - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, shape[0] - 1)
    return indices

def scene_to_results(scene: BasePCOptimizer, min_conf_thr: int,
                     smpl_path: str = "data/citywalkers/full_all_with_measurements.pkl",
                     smpl_model_path: str = "data/smpl",
                     image_dir: str = "data/citywalkers/_fuCbKaSuJ8_77/images",
                     paths: List[str] = []) -> OptimizedResult:
    ### get camera parameters K and T
    K_b33: Float32[np.ndarray, "b 3 3"] = scene.get_intrinsics().numpy(force=True)
    world_T_cam_b44: Float32[np.ndarray, "b 4 4"] = scene.get_im_poses().numpy(
        force=True
    )
    ### image, confidence, depths
    rgb_hw3_list: list[Float32[np.ndarray, "h w 3"]] = scene.imgs
    depth_hw_list: list[Float32[np.ndarray, "h w"]] = [
        depth.numpy(force=True) for depth in scene.get_depthmaps()
    ]
    # normalized depth
    # depth_hw_list = [depth_hw / depth_hw.max() for depth_hw in depth_hw_list]

    conf_hw_list: list[Float32[np.ndarray, "h w"]] = [
        c.numpy(force=True) for c in scene.im_conf
    ]
    # normalize confidence
    # conf_hw_list = [conf_hw / conf_hw.max() for conf_hw in conf_hw_list]

    # point cloud, mesh
    pts3d_list: list[Float32[np.ndarray, "h w 3"]] = [
        pt3d.numpy(force=True) for pt3d in scene.get_pts3d()
    ]
    # get log confidence
    log_conf_trf: Float32[torch.Tensor, ""] = scene.conf_trf(torch.tensor(min_conf_thr))
    # set the minimum confidence threshold
    scene.min_conf_thr = float(log_conf_trf)
    masks_list: Bool[np.ndarray, "h w"] = [
        mask.numpy(force=True) for mask in scene.get_masks()
    ]

    # add smpl meshes
    with open(smpl_path, "rb") as f:
        all_smpl = pickle.load(f)

    smpl_motion = [motion for motion in all_smpl if motion['image'].split('/')[0] == image_dir.split('/')[-1]][0]
    smpl = SMPLLayer(model_path=smpl_model_path)
    tt = lambda x: torch.Tensor(x).float()

    local_orient = smpl_motion['local_orient']
    betas = smpl_motion['betas']
    local_transl = smpl_motion['local_trans']
    body_pose = smpl_motion['body_pose']

    gt_smpl = smpl(body_pose=axis_angle_to_matrix(tt(body_pose)),
                    global_orient=axis_angle_to_matrix(tt(local_orient)),
                    betas=tt(betas),
                    transl=tt(local_transl),
                    pose2rot=False,
                    default_smpl=True)

    joints = vertices2joints(smpl.J_regressor, gt_smpl.vertices).cpu().numpy()

    camera_mat = K_b33[0]

    joints_2d = np.einsum("ij,nj->ni", camera_mat, joints[0])
    joints_2d = joints_2d / joints_2d[:, [2]]
    joints_2d = joints_2d[:, :2]
    joints_2d = get_valid_indices(joints_2d, rgb_hw3_list[0].shape)

    # joints_img = np.zeros_like(rgb_hw3_list[0])
    # joints_img[joints_2d[:, 1].astype(np.int32), joints_2d[:, 0].astype(np.int32)] = 1
    # plt.imshow(joints_img)
    # plt.savefig("debug/joints_img.png")

    J_regressor_feet = torch.from_numpy(np.load(f"{smpl_model_path}/J_regressor_feet.npy")).float()
    feet_joints_all = vertices2joints(J_regressor_feet, gt_smpl.vertices)

    # determine initial scale factor
    smpl_scale_info = []
    all_scales = []
    w, h = rgb_hw3_list[0].shape[1], rgb_hw3_list[0].shape[0]

    for f, frame_path in enumerate(paths): # enumerate(sorted(os.listdir(image_dir))):
        if f == 20: # TODO - hardcoded
            break
        i = int(frame_path[:-4]) - 1
        if i >= len(feet_joints_all):
            continue
        feet_joints = feet_joints_all[i].cpu().numpy()
        feet_joints_2d = np.einsum("ij,nj->ni", camera_mat, feet_joints)
        feet_joints_2d = feet_joints_2d / feet_joints_2d[:, [2]]
        feet_joints_2d = feet_joints_2d[:, :2]
        feet_joints_2d = get_valid_indices(feet_joints_2d, masks_list[f].shape)

        masks_img = masks_list[f].astype(np.int32)
        for r in range(-5, 6):
            for c in range(-5, 6):
                idx_y = feet_joints_2d[[0, 2], 1].astype(np.int32) + r
                idx_x = feet_joints_2d[[0, 2], 0].astype(np.int32) + c
                if (0 <= idx_y).all() and (idx_y < masks_img.shape[0]).all() \
                    and (0 <= idx_x).all() and (idx_x < masks_img.shape[1]).all():
                    masks_img[idx_y, idx_x] = 2
        all_joints_2d = np.einsum("ij,nj->ni", camera_mat, joints[i])
        all_joints_2d = all_joints_2d / all_joints_2d[:, [2]]
        all_joints_2d = all_joints_2d[:, :2]
        all_joints_2d = get_valid_indices(all_joints_2d, masks_img.shape)

        # masks_img[all_joints_2d[:, 1].astype(np.int32), all_joints_2d[:, 0].astype(np.int32)] = 3
        # plt.imshow(masks_img)
        # plt.savefig("debug/mask_img.png")

        for joint_idx in [0, 2]:
            gt_feet_2d = feet_joints_2d[joint_idx]
            if 0 <= gt_feet_2d[0] < w and 0 <= gt_feet_2d[1] < h:
                gt_feet_3d = feet_joints[joint_idx]
                valid_feet_2d = find_closest_point(masks_list[f], gt_feet_2d[1], gt_feet_2d[0])
                # valid_img = masks_list[f].astype(np.int32)
                # for r in range(-5, 6):
                #     for c in range(-5, 6):
                #         idx_y = int(valid_feet_2d[1]) + r
                #         idx_x = int(valid_feet_2d[0]) + c
                #         if 0 <= idx_y < valid_img.shape[0] and 0 <= idx_x < valid_img.shape[1]:
                #             valid_img[idx_y, idx_x] = 2
                # valid_img[all_joints_2d[:, 1].astype(np.int32), all_joints_2d[:, 0].astype(np.int32)] = 3
                # plt.imshow(valid_img)
                # plt.savefig("debug/valid_img.png")

                # depth_img = depth_hw_list[f]
                # plt.imshow(depth_img)
                # plt.savefig("debug/depth_img.png")

                valid_feet_depth = depth_hw_list[f][valid_feet_2d[1], valid_feet_2d[0]]
                scale_factor = gt_feet_3d[2] / valid_feet_depth
                smpl_scale_info.append(torch.tensor([f, valid_feet_2d[1], valid_feet_2d[0], gt_feet_3d[2]]).cuda())
                all_scales.append(scale_factor)

    assert len(all_scales) > 0
    all_scales = np.array(all_scales)
    scale_factor = np.median(all_scales)

    # optimize scale factor with smpl info
    print(f"scale before optimization: {scale_factor}")
    scaled_scene = ScaleOptimizer(scene, torch.stack(smpl_scale_info), scale_factor)
    from mini_dust3r.cloud_opt.optimizer_smpl import PointCloudOptimizerSMPL
    scaled_scene = PointCloudOptimizerSMPL(scene, torch.stack(smpl_scale_info), scale_factor)
    with torch.autograd.set_detect_anomaly(True):
        global_alignment_loop(scaled_scene, niter=150, schedule="linear", lr=0.01)
    # scale_factor = scaled_scene.scale.exp().item()
    scale_factor = scaled_scene.global_scale.exp().item()
    print(f"scale after optimization: {scale_factor}")

    pts3d_list = [pts3d * scale_factor for pts3d in pts3d_list]
    depth_hw_list = [depth * scale_factor for depth in depth_hw_list]

    global_orient = smpl_motion['global_orient']
    local_orient = smpl_motion['local_orient']
    betas = smpl_motion['betas']
    global_transl = smpl_motion['global_trans']
    local_transl = smpl_motion['local_trans']
    body_pose = smpl_motion['body_pose']
    
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

    gt_smpl = smpl(body_pose=tt(body_pose),
            global_orient=tt(global_orient),
            betas=tt(betas),
            transl=tt(transl),
            pose2rot=False,
            default_smpl=True)

    world_T_cam_b44[:, :3, 3] = world_T_cam_b44[:, :3, 3] * scale_factor

    frame_idxs = [int(frame_path[:-4]) - 1 for frame_path in paths]
    # gt_cam_T_world = interpolate_se3(np.linalg.inv(world_T_cam_b44),
    #                          times=np.array(frame_idxs),
    #                          query_times=np.arange(frame_id, frame_idxs[-1] + 1))

    gt_smpl_mesh = []
    for i in range(frame_id, frame_idxs[-1] + 1):
        gt_vertices = gt_smpl.vertices[i]

        # transform to dust3r predicted initial camera frame
        world_T_cam = tt(world_T_cam_b44[0])
        gt_vertices = torch.concatenate([
            gt_vertices, torch.ones(gt_vertices.shape[0], 1)], dim=-1)
        gt_vertices = torch.einsum('ij,nj->ni', world_T_cam, gt_vertices)

        # cam_T_world = tt(gt_cam_T_world[0])
        # gt_vertices = torch.concatenate([
        #     gt_vertices, torch.ones(gt_vertices.shape[0], 1)], dim=-1)
        # gt_vertices = torch.einsum('ij,nj->ni', cam_T_world, gt_vertices)

        gt_smpl_mesh.append(trimesh.Trimesh(vertices=gt_vertices[:, :3], 
                                            faces=smpl.faces))
        
    transl = torch.concatenate([transl[0], torch.ones(1)]).unsqueeze(-1)
    transl = (world_T_cam @ transl).squeeze(-1)

    point_cloud: Float32[np.ndarray, "num_points 3"] = np.concatenate(
        [p[m] for p, m in zip(pts3d_list, masks_list)]
    )
    colors: Float32[np.ndarray, "num_points 3"] = np.concatenate(
        [p[m] for p, m in zip(rgb_hw3_list, masks_list)]
    )

    point_cloud = trimesh.PointCloud(
        point_cloud.reshape(-1, 3), colors=colors.reshape(-1, 3)
    )
    initial_point_cloud = trimesh.PointCloud(pts3d_list[0].reshape(-1, 3), colors=rgb_hw3_list[0].reshape(-1, 3))

    meshes = []
    pbar = tqdm(zip(rgb_hw3_list, pts3d_list, masks_list), total=len(rgb_hw3_list))
    for rgb_hw3, pts3d, mask in pbar:
        meshes.append(pts3d_to_trimesh(rgb_hw3, pts3d, mask))

    mesh = trimesh.Trimesh(**cat_meshes(meshes))
    initial_mesh = trimesh.Trimesh(**meshes[0])

    optimised_result = OptimizedResult(
        K_b33=K_b33,
        world_T_cam_b44=world_T_cam_b44,
        rgb_hw3_list=rgb_hw3_list,
        depth_hw_list=depth_hw_list,
        conf_hw_list=conf_hw_list,
        masks_list=masks_list,
        point_cloud=point_cloud,
        initial_point_cloud=initial_point_cloud,
        mesh=mesh,
        initial_mesh=initial_mesh,
        gt_smpl_mesh=gt_smpl_mesh,
        transl=transl.cpu().numpy()
    )
    return optimised_result


def inference_dust3r(
    image_dir_or_list: Path | list[Path],
    model: AsymmetricCroCo3DStereo,
    device: Literal["cpu", "cuda", "mps"],
    batch_size: int = 1,
    image_size: Literal[224, 512] = 512,
    niter: int = 100,
    schedule: Literal["linear", "cosine"] = "linear",
    min_conf_thr: float = 10,
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
    if isinstance(image_dir_or_list, list):
        imgs, masks, paths = load_images(
            folder_or_list=image_dir_or_list, size=image_size, verbose=True
        )
    elif isinstance(image_dir_or_list, Path):
        imgs, masks, paths = load_images(
            folder_or_list=str(image_dir_or_list), size=image_size, verbose=True
        )
    else:
        raise ValueError("image_dir_or_list should be a list of paths or a path")

    # if only one image was loaded, duplicate it to feed into stereo network
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1

    pairs: list[tuple[ImageDict, ImageDict]] = make_pairs(
        imgs, scene_graph="complete", prefilter=None, symmetrize=True # scene_graph: complete or window-10
    )
    mask_pairs: list[tuple[ImageDict,
                           ImageDict]] = make_pairs(masks,
                                                    scene_graph="complete",
                                                    prefilter=None,
                                                    symmetrize=True)
    output: Dust3rResult = inference(pairs, model, device, batch_size=batch_size)

    for i, (mask1, mask2) in enumerate(mask_pairs):
        output["pred1"]["conf"][i][mask1["img"][0][0] == 0] = 1.0
        output["pred2"]["conf"][i][mask2["img"][0][0] == 0] = 1.0

    # mode = (
    #     GlobalAlignerMode.PointCloudOptimizer
    #     if len(imgs) > 2
    #     else GlobalAlignerMode.PairViewer
    # )
    if len(imgs) <= 2:
        return
    mode = GlobalAlignerMode.PointCloudOptimizer
    scene: BasePCOptimizer = global_aligner(
        dust3r_output=output, device=device, mode=mode, optimize_pp=True
    )

    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="mst", niter=niter, schedule=schedule, lr=lr
        )

    # use camera parameter assumptions
    img_w = 1280
    img_h = 720
    f = (img_h ** 2 + img_w ** 2) ** 0.5
    cx = 0.5 * img_w
    cy = 0.5 * img_h

    cx = cx * output["view1"]['true_shape'][0][1] / img_w
    cy = cy * output["view1"]['true_shape'][0][0] / img_h
    f = f * output["view1"]['true_shape'][0][0] / img_h

    assert np.allclose(output["view1"]['true_shape'][0][1] / img_w,
                       output["view1"]['true_shape'][0][0] / img_h)

    # preset camera intrinsics
    scene.preset_focal(np.array([f]).repeat(len(imgs), axis=0))
    scene.preset_principal_point(
        np.array([[cx, cy]]).repeat(len(imgs), axis=0))

    # get the optimized result from the scene
    optimized_result: OptimizedResult = scene_to_results(scene, min_conf_thr, image_dir=str(image_dir_or_list), paths=paths)
    # optimized_result.point_cloud.export("debug/pointcloud.ply")

    return optimized_result
