# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Main class for the implementation of the global alignment
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn

from mini_dust3r.cloud_opt.base_opt import BasePCOptimizer
from mini_dust3r.cloud_opt.optimizer import PointCloudOptimizer
from mini_dust3r.utils.geometry import xy_grid, geotrf
from mini_dust3r.utils.device import to_cpu, to_numpy


class PointCloudOptimizerSMPL(BasePCOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, old_scene: PointCloudOptimizer, smpl_scale_info, init_scale, *args, 
                optimize_pp=False, 
                focal_break=20, 
                min_conf_thr=3,
                base_scale=1.0,
                allow_pw_adaptors=False,
                pw_break=20,
                rand_pose=torch.randn,
                **kwargs):
        super(BasePCOptimizer, self).__init__(*args, **kwargs)

        self.smpl_scale_info = smpl_scale_info
        # self.smpl_contact_info = smpl_contact_info
        self.smpl_dist = nn.L1Loss()

        # adding thing to optimize
        self.global_scale = nn.Parameter(torch.tensor(np.log(init_scale), dtype=torch.float32))
        self.global_scale.requires_grad_(True)

        # directly copy from the old scenes
        self.edges = old_scene.edges
        self.is_symmetrized = set(self.edges) == {(j, i) for i, j in self.edges}
        self.dist = old_scene.dist
        self.verbose = old_scene.verbose

        self.n_imgs = self._check_edges()
        
        self.pred_i = old_scene.pred_i
        self.pred_j = old_scene.pred_j
        self.imshapes = old_scene.imshapes
        
        self.min_conf_thr = min_conf_thr
        self.conf_trf = old_scene.conf_trf
        self.conf_i = old_scene.conf_i
        self.conf_j = old_scene.conf_j
        self.im_conf = old_scene.im_conf
         
        self.base_scale = base_scale
        self.norm_pw_scale = True
        self.pw_break = pw_break
        self.POSE_DIM = 7
        self.pw_poses =  old_scene.pw_poses  # pairwise poses
        self.pw_adaptors = old_scene.pw_adaptors # slight xy/z adaptation
        self.pw_adaptors.requires_grad_(allow_pw_adaptors)
        self.has_im_poses = False
        self.rand_pose = rand_pose
        self.imgs = old_scene.imgs
        self.has_im_poses = True  # by definition of this class
        self.focal_break = focal_break

        self.im_depthmaps =  old_scene.im_depthmaps  # log(depth)
        self.im_poses = old_scene.im_poses  # camera poses
        self.im_focals = old_scene.im_focals  # camera intrinsics
        self.im_pp = old_scene.im_pp  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)

        self.imshape = self.imshapes[0]
        im_areas = [h*w for h, w in self.imshapes]
        self.max_area = max(im_areas)

        self._pp = old_scene._pp
        self._grid = old_scene._grid
        self._weight_i = old_scene._weight_i
        self._weight_j = old_scene._weight_j
        self._stacked_pred_i = old_scene._stacked_pred_i
        self._stacked_pred_j = old_scene._stacked_pred_j
        self._ei = old_scene._ei
        self._ej = old_scene._ej
        self.total_area_i = old_scene.total_area_i
        self.total_area_j = old_scene.total_area_j

    def _check_all_imgs_are_selected(self, msk):
        assert np.all(self._get_msk_indices(msk) == np.arange(self.n_imgs)), 'incomplete mask!'

    def preset_pose(self, known_poses, pose_msk=None):  # cam-to-world
        self._check_all_imgs_are_selected(pose_msk)

        if isinstance(known_poses, torch.Tensor) and known_poses.ndim == 2:
            known_poses = [known_poses]
        for idx, pose in zip(self._get_msk_indices(pose_msk), known_poses):
            if self.verbose:
                print(f' (setting pose #{idx} = {pose[:3,3]})')
            self._no_grad(self._set_pose(self.im_poses, idx, torch.tensor(pose)))

        # normalize scale if there's less than 1 known pose
        n_known_poses = sum((p.requires_grad is False) for p in self.im_poses)
        self.norm_pw_scale = (n_known_poses <= 1)

        self.im_poses.requires_grad_(False)
        self.norm_pw_scale = False

    def preset_focal(self, known_focals, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, focal in zip(self._get_msk_indices(msk), known_focals):
            if self.verbose:
                print(f' (setting focal #{idx} = {focal})')
            self._no_grad(self._set_focal(idx, focal))

        self.im_focals.requires_grad_(False)

    def preset_principal_point(self, known_pp, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, pp in zip(self._get_msk_indices(msk), known_pp):
            if self.verbose:
                print(f' (setting principal point #{idx} = {pp})')
            self._no_grad(self._set_principal_point(idx, pp))

        self.im_pp.requires_grad_(False)

    def _get_msk_indices(self, msk):
        if msk is None:
            return range(self.n_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == self.n_imgs
            return np.where(msk)[0]
        elif np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            raise ValueError(f'bad {msk=}')

    def _no_grad(self, tensor):
        assert tensor.requires_grad, 'it must be True at this point, otherwise no modification occurs'

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    def get_focals(self):
        log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals])

    def _set_principal_point(self, idx, pp, force=False):
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W/2, H/2)) / 10
        return param

    def get_principal_points(self):
        return self._pp + 10 * self.im_pp

    def get_intrinsics(self):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K

    def get_im_poses(self):  # cam to world
        cam2world = self._get_poses(self.im_poses)
        return cam2world
    
    def get_im_poses_scale(self):  # cam to world
        cam2world = self._get_poses(self.im_poses)
        scaled_translations = cam2world[:, :3, 3] * self.global_scale.exp()
        new_cam2world = cam2world.clone()
        new_cam2world[:, :3, 3] = scaled_translations
        return new_cam2world

    def _set_depthmap(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def get_depthmaps(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res
    
    def get_depthmaps_scale(self, raw=False):
        res = self.im_depthmaps.exp()
        scale = self.global_scale.exp()
        res = res * scale
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
     
        return res

    def depth_to_pts3d(self, global_coord=True):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
       
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)

        # project to world frame
        if global_coord:
            return geotrf(im_poses, rel_ptmaps)
        else: 
            return rel_ptmaps

    def get_pts3d(self, raw=False, global_coord=True):
 
        res = self.depth_to_pts3d(global_coord=global_coord)
 
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res
    
 

    def forward(self):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(raw=True)

        depth_scale = self.get_depthmaps_scale(raw=True)
        poses_scale = self.get_im_poses_scale()

    
        smpl_scale_loss = self.smpl_dist(depth_scale[self.smpl_scale_info[:, 0].long(), 
                                                     self.smpl_scale_info[:, 1].long() * self.imshape[1] + self.smpl_scale_info[:, 2].long()], 
                                                     self.smpl_scale_info[:, 3])
 
        # joint_global_1 = torch.einsum("bij,bj->bi", poses_scale[self.smpl_contact_info[:, 0].long()], self.smpl_contact_info[:, 1:5])
        # joint_global_2 = torch.einsum("bij,bj->bi", poses_scale[self.smpl_contact_info[:, 5].long()], self.smpl_contact_info[:, 6:])
        # smpl_contact_loss = self.smpl_dist(joint_global_1[:, :3], joint_global_2[:, :3])

        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the less
        li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j

        return li + lj + smpl_scale_loss

        # N = len(self.edges)
        # batch_sz = 4
        # li_all, lj_all = 0, 0
        # for i in range(0, N, 4):
        #     # rotate pairwise prediction according to pw_poses
        #     aligned_pred_i = geotrf(pw_poses[i:i+batch_sz], pw_adapt[i:i+batch_sz] * self._stacked_pred_i[i:i+batch_sz])
        #     aligned_pred_j = geotrf(pw_poses[i:i+batch_sz], pw_adapt[i:i+batch_sz] * self._stacked_pred_j[i:i+batch_sz])
        #     # compute the less
        #     li = self.dist(proj_pts3d[self._ei[i:i+batch_sz]], aligned_pred_i, weight=self._weight_i[i:i+batch_sz]).sum() / (self.total_area_i * batch_sz/N)
        #     lj = self.dist(proj_pts3d[self._ej[i:i+batch_sz]], aligned_pred_j, weight=self._weight_j[i:i+batch_sz]).sum() / (self.total_area_j * batch_sz/N)
        #     li_all += li
        #     lj_all += lj
        # return li_all + lj_all



def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)


def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params


def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),)+tensor.shape[1:])))
    return tensor


def acceptable_focal_range(H, W, minf=0.5, maxf=3.5):
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    return minf*focal_base, maxf*focal_base


def apply_mask(img, msk):
    img = img.copy()
    img[msk] = 0
    return img
