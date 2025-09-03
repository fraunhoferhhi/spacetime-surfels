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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import trbfunction

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, iteration = 0, duration=50):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        patch_bbox=viewpoint_camera.random_patch(), # get the whole img
        prcppoint=viewpoint_camera.prcppoint,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        config=pc.config
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    t = viewpoint_camera.timestamp
    pointopacity = pc.get_base_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale

    # equ. (7) in the STG paper
    # trbfdistanceoffset = pc.get_delta_t(viewpoint_camera.timestamp)
    # trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    # trbfoutput = trbfunction(trbfdistance, pc.sgmd_gaussian)
    # opacity = pointopacity * trbfoutput  # - 0.5
    opacity = pc.get_full_opacity(t)

    # (t - mu_i^tilda) in the STG paper
    # tforpoly = trbfdistanceoffset#.detach()

    # equ. (8) in the STG paper
    # explanation: https://github.com/oppo-us-research/SpacetimeGaussians/issues/42#issuecomment-2079837221
    # means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly * tforpoly * tforpoly
    means3D = pc.get_means3D(t)
    # flow3D = pc.get_flow3D(t)
    
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling

        # equ. (9) in the STG paper
        rotations = pc.get_full_rotation(t)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            # TODO: filter with visible_gs_mask
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if pipe.prefilter_for_raster and iteration > 1000:
        # visible_gs_mask = torch.squeeze(opacity > pipe.prefilter_threshold)
        trbfdistanceoffset = pc.get_delta_t(t)
        trbfdistance =  trbfdistanceoffset / torch.exp(pc.get_trbfscale) 
        trbfoutput = trbfunction(trbfdistance, pc.sgmd_gaussian)
        visible_gs_mask = torch.squeeze(trbfoutput > pipe.prefilter_threshold)
        if visible_gs_mask.sum().item() == 0:
            breakpoint()
        means3D = means3D[visible_gs_mask]
        means2D = means2D[visible_gs_mask]
        shs = shs[visible_gs_mask]
        scales = scales[visible_gs_mask]
        rotations = rotations[visible_gs_mask]
        opacity = opacity[visible_gs_mask]
        
        if pipe.convert_SHs_python:
            colors_precomp = colors_precomp[visible_gs_mask]
        if pipe.compute_cov3D_python:
            cov3D_precomp = cov3D_precomp[visible_gs_mask]
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_normal, rendered_depth, rendered_opac, radii,  proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp, 
        )

    if pipe.prefilter_for_raster and iteration > 1000:
        radii_all = radii.new_zeros(visible_gs_mask.shape)
        radii_all[visible_gs_mask] = radii
    else:
        radii_all = radii

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image, 
            "normal": rendered_normal,
            "depth": rendered_depth, 
            "opac": rendered_opac,
            "viewspace_points": screenspace_points, 
            "visibility_filter" : radii_all > 0,
            "radii": radii_all, 
            "gs_per_pixel": gs_per_pixel,
            "weight_per_gs_pixel": weight_per_gs_pixel
            }
