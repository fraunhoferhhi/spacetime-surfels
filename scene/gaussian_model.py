#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, normal2rotation
from torch import nn
import os
from torch.utils.cpp_extension import load
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.general_utils import quaternion2rotmat
from utils.graphics_utils import BasicPointCloud
from utils.image_utils import world2scrn
from utils.general_utils import strip_symmetric, build_scaling_rotation
from mmcv.ops import knn
from utils.general_utils import trbfunction

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, args):
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree  
        self.test_views = args.test_views
        self.sgmd_gaussian = args.sgmd_gaussian
        self.duration = args.duration
        self.prev_num_pts = 0
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.trbf_center_gradient_accum = torch.empty(0)
        self.scale_gradient_accum = torch.empty(0)
        self.opac_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._motion = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._omega = torch.empty(0)
        self.trbfslinit = None
        try:
            self.config = [args.surface, args.normalize_depth, args.perpix_depth]
        except AttributeError:
            self.config = [True, True, True]
        self.setup_functions()
        self.utils_mod = load(name="cuda_utils", sources=["utils/ext.cpp", "utils/cuda_utils.cu"])
        self.motion_degree = args.motion_degree
        self.opac_init = args.opac_init

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.scale_gradient_accum,
            self.opac_accum,
            self.trbf_center_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.config
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        trbf_center_gradient_accum,
        scale_gradient_accum,
        opac_accum,
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.config) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.scale_gradient_accum = scale_gradient_accum
        self.opac_accum = opac_accum
        self.trbf_center_gradient_accum = trbf_center_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_xyz(self):
        return self._xyz # n,3
    
    @property
    def get_trbfcenter(self):
        return self._trbf_center # n,1
    
    @property
    def get_trbfscale(self):
        return self._trbf_scale # n,1
    
    @property
    def get_motion(self):
        return self._motion # n,9
    
    @property
    def get_omega(self):
        return self._omega

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_base_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_base_rotation(self):
        return self.rotation_activation(self._rotation)
    
    def get_full_rotation(self, t):
        delta_t = self.get_delta_t(t)
        rotation =  self._rotation + self._omega * delta_t
        return self.rotation_activation(rotation)
    
    def get_flow3D(self, t):
        delta_t = self.get_delta_t(t)
        flow3D = 0
        for d in range(self.motion_degree):
            flow3D = flow3D + (d+1) * self.get_motion[:,3*d:3*(d+1)] * delta_t ** d
        flow3D *= 1 / self.duration
        return flow3D
    
    def get_means3D(self, t):
        delta_t = self.get_delta_t(t)
        means3D = self.get_xyz
        for d in range(self.motion_degree):
            means3D = means3D + self.get_motion[:,3*d:3*(d+1)] * delta_t ** (d+1)
        return means3D
    
    def get_full_opacity(self, t):
        trbfdistance =  self.get_delta_t(t) / torch.exp(self.get_trbfscale) 
        trbfoutput = trbfunction(trbfdistance, self.sgmd_gaussian)
        opacity = self.get_base_opacity * trbfoutput
        return opacity
    
    def get_delta_t(self, t):
        pointtimes = torch.ones((self.get_xyz.shape[0],1), dtype=self.get_xyz.dtype, requires_grad=False, device="cuda") + 0
        return t * pointtimes - self.get_trbfcenter
    
    def prepare_gs_mask_w_opacity_at_time(self, t, opac_threshold=0.):
        opacity = self.get_full_opacity(t) # n,1
        # mask of gs w. large opacities of a sepecific time
        return torch.squeeze(opacity) > opac_threshold # n
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    # @property
    # def get_normal(self):
    #     return quaternion2rotmat(self.get_full_rotation)[..., 2]


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def interpolate_point(self, pcd: BasicPointCloud):
        oldxyz = pcd.points
        oldcolor = pcd.colors
        oldnormal = pcd.normals
        oldtime = pcd.times
        
        timestamps = np.unique(oldtime)

        newxyz = []
        newcolor = []
        newnormal = []
        newtime = []
        
        for timeidx, time in enumerate(timestamps):
            selectedmask = oldtime == time
            selectedmask = selectedmask.squeeze(1)
            
            if timeidx == 0:
                newxyz.append(oldxyz[selectedmask])
                newcolor.append(oldcolor[selectedmask])
                newnormal.append(oldnormal[selectedmask])
                newtime.append(oldtime[selectedmask])
            else:
                xyzinput = oldxyz[selectedmask]
                xyzinput = torch.from_numpy(xyzinput).float().cuda()
                xyzinput = xyzinput.unsqueeze(0).contiguous() # 1 x N x 3
                xyznnpoints = knn(2, xyzinput, xyzinput, False) # 1 x 2 x N

                nearestneibourindx = xyznnpoints[0, 1].long() # N x 1   
                spatialdistance = torch.norm(xyzinput - xyzinput[:,nearestneibourindx,:], dim=2) #  1 x N
                spatialdistance = spatialdistance.squeeze(0)

                diff_sorted, _ = torch.sort(spatialdistance) 
                N = spatialdistance.shape[0]
                num_take = int(N * 0.25)
                masks = spatialdistance > diff_sorted[-num_take]
                masksnumpy = masks.cpu().numpy()

                newxyz.append(oldxyz[selectedmask][masksnumpy])
                newcolor.append(oldcolor[selectedmask][masksnumpy])
                newnormal.append(oldnormal[selectedmask][masksnumpy])
                newtime.append(oldtime[selectedmask][masksnumpy])
                
        newxyz = np.concatenate(newxyz, axis=0)
        newcolor = np.concatenate(newcolor, axis=0)
        newnormal = np.concatenate(newnormal, axis=0)
        newtime = np.concatenate(newtime, axis=0)
        assert newxyz.shape[0] == newcolor.shape[0]
        
        return BasicPointCloud(points=newxyz, colors=newcolor, normals=newnormal, times=newtime)

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, dup : int = 4):
        print("Number of points total (all frames) : ", len(pcd.points))
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        times = torch.tensor(np.asarray(pcd.times)).float().cuda()

        # (Pdb) unique_elements, counts = np.unique(pcd.times, return_counts=True)
        # (Pdb) unique_elements
        # array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=float32)
        # (Pdb) counts
        # array([2159,  541,  540,  529,  531,  532,  555,  554,  563,  578])

        # distance to nearst neighbor
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2 / 4))[...,None].repeat(1, 3)

        if self.config[0] > 0: # if args.surface is True
            if np.abs(np.sum(pcd.normals)) < 1: # go this way because pcd.normals are all 0
                # duplicate the points to "dup" times with random normals when normals are unknown
                fused_point_cloud = torch.cat([fused_point_cloud for _ in range(dup)], 0)
                fused_color = torch.cat([fused_color for _ in range(dup)], 0)
                scales = torch.cat([scales for _ in range(dup)], 0)
                times = torch.cat([times for _ in range(dup)], 0)
                normals = np.random.rand(len(fused_point_cloud), 3) - 0.5
                normals /= np.linalg.norm(normals, 2, 1, True)
            else:
                normals = pcd.normals

            rots = normal2rotation(torch.from_numpy(normals).to(torch.float32)).to("cuda")
            scales[..., -1] -= 1e10 # squeeze z scaling
        else:
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        opacities = inverse_sigmoid(self.opac_init * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        omega = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        self._omega = nn.Parameter(omega.requires_grad_(True))


        nb_motion_params = 3*self.motion_degree
        motion = torch.zeros((fused_point_cloud.shape[0], nb_motion_params), device="cuda") # if degree=3: x1,x2,x3,  y1,y2,y3, z1,z2,z3
        self._motion = nn.Parameter(motion.requires_grad_(True))

        self._trbf_center = nn.Parameter(times.contiguous().requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True)) 

        if self.trbfslinit is not None:
            nn.init.constant_(self._trbf_scale, self.trbfslinit) # too large ?
        else:
            nn.init.constant_(self._trbf_scale, 0) # too large ?

        nn.init.constant_(self._omega, 0)

    def cache_gradient(self):
        self._xyz_grd += self._xyz.grad.clone()
        self._features_dc_grd += self._features_dc.grad.clone()
        self._features_rest_grd += self._features_rest.grad.clone()
        self._scaling_grd += self._scaling.grad.clone()
        self._rotation_grd += self._rotation.grad.clone()
        self._opacity_grd += self._opacity.grad.clone()
        self._trbf_center_grd += self._trbf_center.grad.clone()
        self._trbf_scale_grd += self._trbf_scale.grad.clone()
        if self.motion_degree > 0: self._motion_grd += self._motion.grad.clone()
        self._omega_grd += self._omega.grad.clone()
    
    def zero_gradient_cache(self):

        self._xyz_grd = torch.zeros_like(self._xyz, requires_grad=False)
        self._features_dc_grd = torch.zeros_like(self._features_dc, requires_grad=False)
        self._features_rest_grd = torch.zeros_like(self._features_rest, requires_grad=False)
        self._scaling_grd = torch.zeros_like(self._scaling, requires_grad=False)
        self._rotation_grd = torch.zeros_like(self._rotation, requires_grad=False)
        self._opacity_grd = torch.zeros_like(self._opacity, requires_grad=False)
        self._trbf_center_grd = torch.zeros_like(self._trbf_center, requires_grad=False)
        self._trbf_scale_grd = torch.zeros_like(self._trbf_scale, requires_grad=False)
        self._motion_grd = torch.zeros_like(self._motion, requires_grad=False)
        self._omega_grd = torch.zeros_like(self._omega, requires_grad=False)


    def set_batch_gradient(self, cnt):
        ratio = 1/cnt
        self._features_dc.grad = self._features_dc_grd * ratio
        self._features_rest.grad = self._features_rest_grd * ratio
        self._xyz.grad = self._xyz_grd * ratio
        self._scaling.grad = self._scaling_grd * ratio
        self._rotation.grad = self._rotation_grd * ratio
        self._opacity.grad = self._opacity_grd * ratio
        self._trbf_center.grad = self._trbf_center_grd * ratio
        self._trbf_scale.grad = self._trbf_scale_grd* ratio
        self._motion.grad = self._motion_grd * ratio
        self._omega.grad = self._omega_grd * ratio

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opac_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.trbf_center_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._omega], 'lr': training_args.omega_lr, "name": "omega"},
            {'params': [self._trbf_center], 'lr': training_args.trbfc_lr, "name": "trbf_center"},
            {'params': [self._trbf_scale], 'lr': training_args.trbfs_lr, "name": "trbf_scale"},
            {'params': [self._motion], 'lr':  training_args.position_lr_init * self.spatial_lr_scale * 0.5 * training_args.movelr , "name": "motion"},
        ]

        self.config[3] = training_args.camera_lr > 0

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult, # useless because lr_delay_steps == 0
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration, training_args):
        ''' Learning rate scheduling per step '''
        
        def get_expon_lr_func_out(lr_init, final_ratio = 0.01):
            return get_expon_lr_func(lr_init=lr_init, lr_final=lr_init * final_ratio, max_steps=training_args.iterations)
        
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            
            elif param_group["name"] == "trbf_center":
                lr_func = get_expon_lr_func_out(training_args.trbfc_lr)
                param_group['lr'] = lr_func(iteration)
            elif param_group["name"] == "trbf_scale":
                lr_func = get_expon_lr_func_out(training_args.trbfs_lr)
                param_group['lr'] = lr_func(iteration)
            elif param_group["name"] == "omega":
                lr_func = get_expon_lr_func_out(training_args.omega_lr)
                param_group['lr'] = lr_func(iteration)
            elif param_group["name"] == "motion":
                lr_func = get_expon_lr_func_out(training_args.position_lr_init * self.spatial_lr_scale * 0.5 * training_args.movelr)
                param_group['lr'] = lr_func(iteration)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'trbf_center', 'trbf_scale','nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._motion.shape[1]):
            l.append('motion_{}'.format(i))
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._omega.shape[1]):
            l.append('omega_{}'.format(i))
        
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        trbf_center= self._trbf_center.detach().cpu().numpy()
        trbf_scale = self._trbf_scale.detach().cpu().numpy()
        motion = self._motion.detach().cpu().numpy()
        omega = self._omega.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, trbf_center, trbf_scale, normals, motion, f_dc, f_rest, opacities, scale, rotation, omega), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, ratio):
        # opacities_new = inverse_sigmoid(torch.min(self.get_base_opacity, torch.ones_like(self.get_base_opacity) * ratio))
        opacities_new = inverse_sigmoid(self.get_base_opacity * ratio)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        trbf_center= np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(plydata.elements[0]["trbf_scale"])[..., np.newaxis]

        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion")]
        nummotion = 3*self.motion_degree
        motion = np.zeros((xyz.shape[0], nummotion))
        for i in range(nummotion):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_"+str(i)])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omega_names = sorted(omega_names, key = lambda x: int(x.split('_')[-1]))
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_center = nn.Parameter(torch.tensor(trbf_center, dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.tensor(trbf_scale, dtype=torch.float, device="cuda").requires_grad_(True))
        self._motion = nn.Parameter(torch.tensor(motion, dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.scale_gradient_accum = self.scale_gradient_accum[valid_points_mask]
        self.opac_accum = self.opac_accum[valid_points_mask]
        self.trbf_center_gradient_accum = self.trbf_center_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_trbf_center, new_trbfscale, new_motion, new_omega, dummy=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "trbf_center" : new_trbf_center,
        "trbf_scale" : new_trbfscale,
        "motion": new_motion,
        "omega": new_omega,}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opac_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.trbf_center_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, pre_mask=True):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # print(selected_pts_mask.dtype, pre_mask.dtype)
        # selected_pts_mask *= pre_mask
        
        # print(f'n_split: {selected_pts_mask.sum()}')

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # if self.config[0] > 0: # if args.surface is True
        #     new_scaling[:, -1] = -1e10
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N,1)
        # new_trbf_center = torch.rand_like(new_trbf_center) #* 0.5
        new_trbf_scale = self._trbf_scale[selected_pts_mask].repeat(N,1)
        new_motion = self._motion[selected_pts_mask].repeat(N,1)
        new_omega = self._omega[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_trbf_center, new_trbf_scale, new_motion, new_omega)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, pre_mask=True):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        selected_pts_mask *= pre_mask
        # print(f'n_clone: {selected_pts_mask.sum()}')
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # new_trbf_center =  torch.rand((self._trbf_center[selected_pts_mask].shape[0], 1), device="cuda")  #self._trbf_center[selected_pts_mask]
        new_trbf_center =  self._trbf_center[selected_pts_mask]
        new_trbfscale = self._trbf_scale[selected_pts_mask]
        new_motion = self._motion[selected_pts_mask]
        new_omega = self._omega[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_trbf_center, new_trbfscale, new_motion, new_omega)

    def adaptive_prune(self, min_opacity, extent):
        n_ori = len(self._xyz)
        prune_opac = (self.get_base_opacity < min_opacity).squeeze()

        # scale_thrsh = torch.tensor([2e-4, 0.1]) * extent
        scale_min = self.get_scaling[:, :2].min(1).values
        scale_max = self.get_scaling[:, :2].max(1).values
        prune_scale = scale_max > 0.5 * extent
        prune_scale += (scale_min * scale_max) < (1e-8 * extent**2)
        
        prune_vis = (self.denom == 0).squeeze()

        # prune pionts w. time range shorter than 1 frame: 
        # 2 * std_dev < 1 / n_frames(duration)
        # e^trbf_scale < 1 / 2duration
        # trbf_scale < ln (1 / 2duration)
        # prune_time_range = self.get_trbfscale.squeeze() < -4

        prune = prune_opac + prune_vis + prune_scale # + prune_time_range
        self.prune_points(prune)
        # print(f'opac:{prune_opac.sum()}, scale:{prune_scale.sum()}, vis:{prune_vis.sum()} extend:{extent}')
        # print(f'prune: {n_ori}-->{len(self._xyz)}')
        
        # log values:
        nb_prune_opac = prune_opac.float().sum()
        nb_prune_scale = prune_scale.float().sum()
        nb_prune_vis = prune_vis.float().sum()
        return {
                "nb_prune_opac": nb_prune_opac,
                "nb_prune_scale": nb_prune_scale,
                "nb_prune_vis": nb_prune_vis,
                }
        

    def knn_prune(self, knn_prune_ratio=0.25):
        xyzinput = self.get_xyz
        xyzinput = xyzinput.unsqueeze(0).contiguous() # 1 x N x 3
        xyznnpoints = knn(2, xyzinput, xyzinput, False) # 1 x 2 x N

        nearestneibourindx = xyznnpoints[0, 1].long() # N x 1   
        spatialdistance = torch.norm(xyzinput - xyzinput[:,nearestneibourindx,:], dim=2) #  1 x N
        spatialdistance = spatialdistance.squeeze(0)

        diff_sorted, _ = torch.sort(spatialdistance) 
        N = spatialdistance.shape[0]
        num_take = int(N * knn_prune_ratio)
        prune = spatialdistance > diff_sorted[-num_take]
        self.prune_points(prune)
        
    
    def adaptive_densify(self, max_grad, extent):
        grad_pos = self.xyz_gradient_accum / self.denom
        # grad_scale = self.scale_gradient_accum /self.denom
        # running_avg_opac = self.opac_accum /self.denom
        grad_pos[grad_pos.isnan()] = 0.0
        # grad_scale[grad_scale.isnan()] = 0.0
        # running_avg_opac[running_avg_opac.isnan()] = 0.0

        # larger = torch.le(grad_scale, 1e-7)[:, 0] #if opac_lr == 0 else True
        # denser = torch.le(running_avg_opac, 2)[:, 0]
        # pre_mask = denser * larger

        grad_t_center = self.trbf_center_gradient_accum / self.denom
        
        self.densify_and_clone(grad_pos, max_grad, extent)
        self.densify_and_split(grad_pos, max_grad, extent)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # gradients of projected 2d gausssian
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # gradients of 3d scene gausssian
        self.scale_gradient_accum[update_filter] += self._scaling.grad[update_filter, :2].sum(1, True)
        self.opac_accum[update_filter] += self._opacity[update_filter]
        self.trbf_center_gradient_accum[update_filter] += torch.norm(self._trbf_center.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def to_occ_grid(self, cutoff, grid_dim_max=512, bound_overwrite=None, timestamp=0.0, opac_thrsh_4_occ_grid=0.):
        xyz = self._xyz #get_xyz
        gs_mask = self.prepare_gs_mask_w_opacity_at_time(timestamp, opac_thrsh_4_occ_grid)
        opacity = self.get_full_opacity(timestamp)
        means3D = self.get_means3D(timestamp)
        rotations = self.get_full_rotation(timestamp)
        
        if bound_overwrite is None:
            xyz_min = (self._xyz[gs_mask]).min(0)[0]
            xyz_max = (self._xyz[gs_mask]).max(0)[0]
            xyz_len = xyz_max - xyz_min
            xyz_min -= xyz_len * 0.1
            xyz_max += xyz_len * 0.1
        else:
            xyz_min, xyz_max = bound_overwrite
        xyz_len = xyz_max - xyz_min

        # voxel size
        grid_len = xyz_len.max() / grid_dim_max
        # necessary grid size, smaller than [512,512,512]
        grid_dim = (xyz_len / grid_len + 0.5).to(torch.int32)

        grid = self.utils_mod.gaussian2occgrid(xyz_min, xyz_max, grid_len, grid_dim,
                                               means3D[gs_mask], rotations[gs_mask], (self.get_scaling)[gs_mask], opacity[gs_mask],
                                               torch.tensor([cutoff]).to(torch.float32).cuda())
            
        return grid, -xyz_min, 1 / grid_len, grid_dim