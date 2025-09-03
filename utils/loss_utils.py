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
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from utils.general_utils import knn_pcl
# from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
from gaussian_renderer import render
from utils.flow_utils import project, pixel_to_gaussian_idx, compute_fwdbwd_mask
from utils.time_logger import DummyTimeLogger
import math


def l1_loss(network_output, gt, weight=1):
    return torch.abs((network_output - gt)).mean()

def cos_loss(output, gt, thrsh=0, weight=1):
    cos = torch.sum(output * gt * weight, 0)
    return (1 - cos[cos < np.cos(thrsh)]).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def l1_loss_weighted(network_output, gt, weight):
    return torch.abs((network_output - gt) * weight).mean()

def l2_loss_weighted(network_output, gt, weight):
    return (((network_output - gt) ** 2) * weight).mean()

def bce_loss(output, mask=1):
    bce = output * torch.log(output) + (1 - output) * torch.log(1 - output)
    loss = (-bce * mask).mean()
    return loss

def compute_flow_loss(opt, gaussians, viewpoint_cam, render_pkg, pipe, background):
    """
    Computes the optical flow loss for Gaussian-based motion optimization in a 3D scene.

    Parameters:
        opt (object): Options containing configuration flags and thresholds.
        pipe (object): Rendering pipeline for processing Gaussian splats.
        gaussians (object): Gaussian parameters (positions, scaling, rotation, etc.).
        viewpoint_cam (object): Current viewpoint camera (projection matrices, flow ground truth, etc.).
        psnr_ (float): Current PSNR value for deciding flow optimization thresholds.
        H (int): Image height.
        W (int): Image width.
        background (torch.Tensor): Background color tensor.
        render_pkg (dict): Package containing pre-rendered results (e.g., depth map, Gaussian per-pixel data).

    Returns:
        loss (float): Computed flow loss to be backpropagated.
    """
    W = viewpoint_cam.image_width
    H = viewpoint_cam.image_height
    # Retrieve ground truth optical flow (forward and backward) and occlusion maps.
    gt_flow, gt_flow_bwd = viewpoint_cam.get_gtFlow()
    gt_flow, gt_flow_bwd = (d.cuda() if d is not None else None for d in (gt_flow, gt_flow_bwd))
    
    if gt_flow is not None and gt_flow_bwd is not None:
        flow_mask, flow_mask_bwd = compute_fwdbwd_mask(gt_flow, gt_flow_bwd)
    else:
        # Handle missing occlusion maps by initializing them with zeros.
        flow_mask = torch.zeros([H, W]).bool().cuda()
        flow_mask_bwd = torch.zeros([H, W]).bool().cuda()
    
    w2c = viewpoint_cam.w2c  # World-to-camera transformation.
    k = viewpoint_cam.K     # Intrinsic camera matrix.

    is_fg = torch.ones_like(gaussians.get_base_opacity).bool().squeeze(-1)

    # Select the Gaussians contributing to the flow loss.
    # Extract depth map and randomly sample pixels for flow supervision.
    depth = render_pkg["depth"]
    opac = render_pkg["opac"]
    mask_gt = viewpoint_cam.get_gtMask(True) > 0
    mask_vis = (opac.detach() > 1e-1) #1e-5
    mask = mask_vis * mask_gt

    seg_map = (torch.rand_like(depth).squeeze(0) <= opt.flow_sample_ratio)
    seg_map &= (flow_mask & flow_mask_bwd & mask.squeeze())  # Exclude occluded regions.

    render_fn = lambda flow2render: render(
        viewpoint_camera=viewpoint_cam,
        pc=gaussians,
        pipe=pipe,
        bg_color=background,
        override_color=flow2render,
    )

    # Ensure at least one pixel is sampled to avoid empty masks.
    if not seg_map.any():
        seg_map[seg_map.shape[0] // 2, seg_map.shape[1] // 2] = True

    if opt.soft_select_fg:
        if "gs_per_pixel" in render_pkg:
            # Select Gaussians per pixel from render package.
            fg_gaussian_idx = (render_pkg["gs_per_pixel"][:, seg_map]).T
            fg_gaussian_contrib = (render_pkg["weight_per_gs_pixel"][:, seg_map]).T.unsqueeze(-1)
            fg_gaussian_idx = fg_gaussian_idx.to(torch.int64)
        else:
            # Find k nearest Gaussians for motion optimization.
            fg_gaussian_idx, fg_gaussian_contrib = pixel_to_gaussian_idx(
                depth, w2c, k, seg_map, gaussians, is_fg, K=opt.flow_k, return_K=True, timestamp=viewpoint_cam.timestamp
            )
        
        fg_shape = fg_gaussian_idx.shape # (H*W, K)
        fg_gaussian_contrib = torch.nan_to_num(fg_gaussian_contrib, posinf=torch.finfo(fg_gaussian_contrib.dtype).max)
        # Normalize Gaussian contributions to ensure stable optimization.
        # fg_gaussian_contrib_max = fg_gaussian_contrib.max(1, keepdim=True).values # for each pixel, pick max weight out of the k nearest neighbors
        # Sum across the K nearest neighbors dimension
        fg_gaussian_contrib_sum = fg_gaussian_contrib.sum(1, keepdim=True)  # (H*W, 1, 1)
        fg_gaussian_contrib = fg_gaussian_contrib / (fg_gaussian_contrib_sum + 1e-8)
        fg_gaussian_idx = fg_gaussian_idx.reshape(-1) # (H*W*K)

        # # Optionally adjust contributions with predicted confidence values.
        # if opt.predict_confidence:
        #     confidence = render_pkg["confidence"][fg_gaussian_idx].reshape(*fg_gaussian_idx.shape, 1)
        #     fg_gaussian_contrib = fg_gaussian_contrib * confidence
        
        # Define the flow loss function for soft selected Gaussians.
        flow_loss_fn = lambda pred, gt: l1_loss_weighted( 
            pred.reshape(-1, opt.flow_k, 2), # (H*W*K, 2) -> (H*W, K, 2)
            gt.reshape(-1, 2).unsqueeze(1).repeat(1, opt.flow_k, 1), # (2, H, W) -> (H*W, K, 2) (repeat K times)
            weight=fg_gaussian_contrib
        )
    else:
        # Select the Gaussian with the maximum contribution for each pixel.
        if "gs_per_pixel" in render_pkg:
            fg_gaussian_idx = render_pkg["gs_per_pixel"][:, seg_map].T
            fg_gaussian_top_idx = (render_pkg["weight_per_gs_pixel"][:, seg_map]).T.argmax(-1).unsqueeze(-1)
            fg_gaussian_idx = torch.gather(fg_gaussian_idx, dim=1, index=fg_gaussian_top_idx).squeeze(-1)
        else:
            fg_gaussian_idx = pixel_to_gaussian_idx(depth, w2c, k, seg_map, gaussians, is_fg, K=opt.flow_k, return_K=False, timestamp=viewpoint_cam.timestamp)
        
        # Define a standard smooth L1 loss function.
        flow_loss_fn = F.smooth_l1_loss

    flow_fwd_per_gs = None
    flow_bwd_per_gs = None

    if gt_flow is not None:
        current_mean3D = gaussians.get_means3D(viewpoint_cam.timestamp)[fg_gaussian_idx] # position at t, Shape: (H*W*K, 3)
        next_mean3D = gaussians.get_means3D(viewpoint_cam.timestamp + 1/gaussians.duration)[fg_gaussian_idx] # position at t+1, Shape: (H*W*K, 3)
        curr_uv, _ = project(current_mean3D.T, w2c, k) # Shape: (2, H*W*K)
        next_uv, _ = project(next_mean3D.T, w2c, k) # Shape: (2, H*W*K)
        flow_fwd_per_gs = -(next_uv - curr_uv).T # Shape: (2, H*W*K)
        # Reshape the flow to group by pixels and Gaussian components
        H = gt_flow.shape[1]
        W = gt_flow.shape[2]
        # flow_fwd_reshaped = flow_fwd_per_gs.view(H, W, opt.flow_k, 2)  # Shape: (H, W, K, 2)
        # weights_reshaped = fg_gaussian_contrib.view(H, W, opt.flow_k, 1)  # Shape: (H, W, K, 1)

        # # Compute the weighted sum of 3D flows across Gaussian components for each pixel
        # flow_fwd_per_pix = (flow_fwd_reshaped * weights_reshaped).sum(dim=2)  # Shape: (H, W, 2)
        # flow_fwd_loss = l1_loss_weighted(flow_fwd_per_pix.permute(2, 0, 1), gt_flow, weight=1)
        flow_fwd_loss = flow_loss_fn(flow_fwd_per_gs.T, gt_flow[:, seg_map])
        loss = flow_fwd_loss

        if gt_flow_bwd is not None:
            flow_bwd_per_gs = -(curr_uv - next_uv).T
            # flow_bwd_reshaped = flow_bwd_per_gs.view(H, W, opt.flow_k, 2)  # Shape: (H, W, K, 2)
            # flow_bwd_per_pix = (flow_bwd_reshaped * weights_reshaped).sum(dim=2)  # Shape: (H, W, 2)
            # flow_bwd_loss = l1_loss_weighted(flow_bwd_per_pix.permute(2, 0, 1), gt_flow_bwd, weight=1)
            flow_bwd_loss = flow_loss_fn(flow_bwd_per_gs.T, gt_flow_bwd[:, seg_map])
            loss += flow_bwd_loss
            
        return loss, gt_flow, gt_flow_bwd, seg_map, flow_fwd_per_gs, flow_bwd_per_gs, fg_gaussian_contrib 
    
    return torch.tensor(0, device='cuda:0'), None, None, None, None, None, None

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def knn_smooth_loss(gaussian, K):
    xyz = gaussian._xyz
    normal = gaussian.get_normal
    nn_xyz, nn_normal = knn_pcl(xyz, normal, K)
    dist_prj = torch.sum((xyz - nn_xyz) * normal, -1, True).abs()
    loss_prj = dist_prj.mean()

    nn_normal = torch.nn.functional.normalize(nn_normal)
    loss_normal = cos_loss(normal, nn_normal, thrsh=np.pi / 3)
    return loss_prj, loss_normal

# Taken from MAGS: https://github.com/jasongzy/MAGS/blob/main/utils/flow_utils.py#L479
class CosineAnnealing:
    def __init__(self, warmup_step: int, total_step: int, max_value=1.0, min_value=0.0):
        self.warmup_step = warmup_step
        self.total_step = total_step
        self.max_value = max_value
        self.min_value = min_value
        self.enable = True

    def _linear(self, step: int):
        k = (self.max_value - self.min_value) / self.warmup_step
        return k * step + self.min_value

    def _cosine(self, step: int):
        return self.min_value + 0.5 * (self.max_value - self.min_value) * (
            1 + math.cos(math.pi * (step - self.warmup_step) / (self.total_step - self.warmup_step))
        )

    def get_value(self, step: int, decay=True):
        assert step >= 0, "step must be non-negative"
        if self.enable:
            if decay:
                if step <= self.warmup_step:
                    return self._linear(step)
                elif step <= self.total_step:
                    return self._cosine(step)
                else:
                    return self.min_value
            else:
                if step <= self.warmup_step:
                    return self._linear(step)
                else:
                    return self.max_value
        else:
            return self.max_value

    def __call__(self, *args, **kwargs):
        return self.get_value(*args, **kwargs)