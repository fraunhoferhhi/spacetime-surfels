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

import os
import torch
from random import randint
import random 
from utils.loss_utils import l1_loss, ssim, cos_loss, l2_loss, compute_flow_loss
from gaussian_renderer import render
import numpy as np
import re
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, setgtisint8
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, depth2rgb, normal2rgb, depth2normal, normal2curv, masked_psnr, resize_image
from utils.extra_utils import get_high_velocity_frames, o3d_knn, knn_self_pytorch_batched
from utils.time_logger import TimeLogger
from utils.flow_utils import get_flow_images
from utils.loss_utils import CosineAnnealing
from torchvision.utils import save_image
from argparse import ArgumentParser, Namespace
import time
import os
import json
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args, load_config_file
from submodules.RAFT.utils import flow_viz
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(args):
    # tb_writer = prepare_output_and_logger(dataset)
    tb_writer = prepare_output_and_logger(args)
    
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    
    opt.position_lr_max_steps = args.iterations
    opt.densify_until_iter = args.iterations / 2

    first_iter = 0
    use_mask = dataset.use_mask
    
    gaussians = GaussianModel(dataset)
    gaussians.trbfslinit = opt.trbfslinit
    
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if args.checkpoint:
        (model_params, first_iter) = torch.load(args.checkpoint)
        gaussians.restore(model_params, opt)
    elif use_mask: # visual hull init
        # gaussians.mask_prune(scene.getTrainCameras(), 4)
        None

    nb_cams_static = int(len(scene.getTrainCameras())/dataset.duration)
    opt.densification_interval = max(opt.densification_interval, nb_cams_static)

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    time_loggers = {
        "preparation": TimeLogger(),
        "camera_sampling": TimeLogger(),
        "rendering": TimeLogger(),
        "pre_loss_computation": TimeLogger(),
        "loss_computation": TimeLogger(),
        "backward_pass": TimeLogger(),
        "optimizer_step": TimeLogger(),
        "densification": TimeLogger(),
        "write_progress": TimeLogger(),
        "checkpoint_save": TimeLogger(),
        "gaussian_save": TimeLogger(),
        "training_report": TimeLogger(),
        "total_iter_time": TimeLogger()
    }
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    weight_scheduler = CosineAnnealing(warmup_step=opt.flow_warmup_step, total_step=opt.iterations, max_value=1.0, min_value=0.0)
    weight_scheduler.enable = opt.flow_weight_decay
    for iteration in range(first_iter, opt.iterations + 1):
        with time_loggers["total_iter_time"].time_block():
            
            # Setup
            with time_loggers["preparation"].time_block():
                iter_start.record()
                gaussians.update_learning_rate(iteration, opt)
                
                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                if (iteration - 1) == args.debug_from:
                    pipe.debug = True
                
                background = torch.rand((3), dtype=torch.float32, device="cuda") if dataset.random_background else background
            
            if opt.batch > 1: gaussians.zero_gradient_cache()

            for i in range(opt.batch):
                if not viewpoint_stack:
                    if opt.velocity_based_sampling and iteration > 10000:
                        # Determine whether to use high-velocity frames or all frames
                        views_per_timestep = len(scene.getTrainCameras()) // dataset.duration
                        high_velocity_stack_size = int(dataset.duration * 0.1 * views_per_timestep)  # 10% of frames (because we use 90th percentile for high velocity frame selection)
                        uniform_sampling_iterations = high_velocity_stack_size // 3  # 3:1 ratio
                        
                        # Use high-velocity frames if we're not in the uniform sampling phase
                        if iteration % (high_velocity_stack_size + uniform_sampling_iterations) < high_velocity_stack_size:
                            high_velocity_time_indices = get_high_velocity_frames(dataset, pipe, gaussians) / dataset.duration
                            viewpoint_stack = [
                                cam for cam in scene.getTrainCameras() 
                                if cam.timestamp in high_velocity_time_indices
                            ]
                            if not viewpoint_stack:  # Fallback if no high velocity frames found
                                viewpoint_stack = scene.getTrainCameras().copy()
                        else:
                            # Use all frames for uniform_sampling_iterations
                            viewpoint_stack = scene.getTrainCameras().copy()
                    else:
                        viewpoint_stack = scene.getTrainCameras().copy()

                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

                
                # Rendering
                with time_loggers["rendering"].time_block():
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background, duration=dataset.duration)
                
                # Prepare data for loss computation
                with time_loggers["pre_loss_computation"].time_block():    
                    image, normal, depth, opac, viewspace_point_tensor, visibility_filter, radii = \
                    render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
                    render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    
                    gt_mask = viewpoint_cam.get_gtMask(use_mask)
                    gt_image = viewpoint_cam.get_gtImage(background, use_mask)
                    mask_vis = (opac.detach() > 1e-5)
                    normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis
                    d2n = depth2normal(depth, mask_vis, viewpoint_cam)
                    mono = viewpoint_cam.mono if dataset.mono_normal else None
                    if mono is not None:
                        mono *= gt_mask
                        monoN = mono[:3]
                        
                    # gt_flow_fwd, _ = viewpoint_cam.get_gtFlow() if dataset.use_flow else (None, None)

                # Loss computation
                with time_loggers["loss_computation"].time_block():
                    loss_dict = {}
                    loss_l1 = l1_loss(image, gt_image)
                    psnr_ = psnr(image, gt_image).mean().double()
                    loss_rgb = (1.0 - opt.lambda_dssim) * loss_l1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                    loss_dict["loss_rgb"] = loss_rgb

                    bce_loss_func = torch.nn.BCELoss()
                    loss_mask = bce_loss_func(opac, gt_mask)
                    loss_dict["loss_mask"] = loss_mask * opt.lambda_mask
                    
                    if opt.opac_loss == "gsurfel":
                        # original gsurfel (right half of a Normal distribution)
                        opac_ = gaussians.get_base_opacity
                        opac_mask = torch.gt(opac_, 0.51) * torch.le(opac_, 0.99)
                        loss_opac = (torch.exp(-(opac_ - 0.5)**2 * 20) * opac_mask).mean()
                    elif opt.opac_loss == "normal":
                        # original gsurfel (full Normal distribution)
                        opac_ = gaussians.get_base_opacity
                        opac_mask = torch.gt(opac_, 0.01) * torch.le(opac_, 0.99)
                        loss_opac = (torch.exp(-(opac_ - 0.5)**2 * 20) * opac_mask).mean()
                    elif opt.opac_loss == "sigmoid":
                        # sigmoid
                        opac_ = gaussians.get_base_opacity
                        opac_mask = torch.gt(opac_, 0.01) * torch.le(opac_, 0.99)
                        loss_opac = (torch.sigmoid(-10 * (opac_ - 0.5)) * opac_mask).mean()
                    elif opt.opac_loss == "softplus":
                        # softplus
                        opac_ = gaussians.get_base_opacity
                        opac_mask = torch.gt(opac_, 0.01) * torch.le(opac_, 0.99)
                        loss_opac = (torch.nn.functional.softplus(-10 * (opac_ - 0.5)) * opac_mask).mean()
                    elif opt.opac_loss == "redfunction":
                        # # accidently found (red function)
                        opac_ = gaussians.get_base_opacity
                        opac_mask = torch.gt(opac_, 0.01) * torch.le(opac_, 0.99)
                        loss_opac = (torch.exp(-(opac_ - 0.5) * 10) * opac_mask).mean()
                    else:
                        raise NotImplementedError("Unknown opacity loss")
                    
                    loss_dict["loss_opac"] = loss_opac * opt.lambda_opac
                    
                    # smoothness loss on motion and rotation coefficients. Adjusted from E-D3DGS
                    # if opt.lambda_smooth_coeff > 0:
                    #     if gaussians.prev_num_pts != gaussians._xyz.shape[0] or (iteration > opt.densify_until_iter and iteration % opt.coeff_smooth_reset_interval == 0):
                    if opt.lambda_smooth_coeff > 0 and iteration < opt.densify_until_iter:
                        if gaussians.prev_num_pts != gaussians._xyz.shape[0]:
                            gaussians.prev_num_pts = gaussians._xyz.shape[0]
                            k = 20
                            
                            ## Brianne's implementation
                            # nn_sq_dist, nn_indices = o3d_knn(gaussians.get_xyz.detach().cpu().numpy(), k)
                            # nn_weight = np.exp(-2000 * nn_sq_dist)
                            # nn_indices = torch.tensor(nn_indices).cuda().long().contiguous()
                            # nn_spatial_weight = torch.tensor(nn_weight).cuda().float().contiguous()
                            
                            # trbf_center = gaussians.get_trbfcenter.detach().contiguous()
                            # trbf_center_rep = trbf_center[:, None, :].repeat(1, k, 1)  # Broadcast for each neighbor
                            # trbf_scale = gaussians.get_trbfscale.detach().contiguous()
                            # trbf_scale_rep = trbf_scale[:, None, :].repeat(1, k, 1)    # Broadcast for each neighbor
                            # std_dev = torch.exp(trbf_scale_rep) # Convert to std_Dev
                            # std_dev = std_dev.clamp(min=1e-5)
                            # trbf_center_knn = trbf_center[nn_indices].detach().contiguous()
                            
                            # temporal_diff = (trbf_center_rep - trbf_center_knn).abs()
                            # nn_temporal_weight = torch.exp(-((temporal_diff / std_dev) ** 2)).squeeze()  # Gaussian-like weighting
                            # nn_weight = nn_spatial_weight * nn_temporal_weight
                        
                            # Decai's implementation
                            # rescale time dimension because we don't want nn across different time steps so much?
                            xyzt = torch.cat([gaussians.get_xyz, gaussians.get_trbfcenter * 10], dim=1).detach().contiguous() # N, 4
                            nn_sq_dists, nn_indices = knn_self_pytorch_batched(xyzt, k, batch_size=1024*4) # [N, k], [N, k]
                            nn_weight = torch.exp(-2000 * nn_sq_dists).contiguous() # [N, k]

                        coeffs = torch.cat((gaussians._motion, gaussians._omega), dim=1) # [N, C/13]
                        coeffs_repeat = coeffs[:,None,:].repeat(1,k,1) # [N, k, C/13]
                        coeffs_knn = coeffs[nn_indices] # [N, k, C/13]
                        loss_smooth_coeff = torch.sqrt(((coeffs_knn - coeffs_repeat) ** 2).sum(-1) * nn_weight + 1e-20).mean()
                        # loss_dict["loss_smooth_coeff"] = (1 - iteration / opt.iterations * 1) * loss_smooth_coeff * opt.lambda_smooth_coeff # decaying
                        loss_dict["loss_smooth_coeff"] = loss_smooth_coeff * opt.lambda_smooth_coeff
                    else:
                        loss_dict["loss_smooth_coeff"] = torch.tensor(0, device='cuda:0')

                    # gaussian shape regularization loss
                    min_scale = gaussians.get_scaling[:,:2].min()
                    max_scale = gaussians.get_scaling[:,:2].max()
                    loss_shape_reg = torch.log(max_scale / min_scale) 
                    loss_dict["loss_shape_reg"] = loss_shape_reg * opt.lambda_shape_reg
                    
                    # flow loss
                    # if dataset.use_flow and gt_flow_fwd is not None:
                    #     gt_flow_fwd = gt_flow_fwd.to('cuda:0')
                    #     loss_flow = l1_loss(flow, gt_flow_fwd)
                    # else:
                    #     loss_flow = torch.tensor(0, device='cuda:0')
                    # loss_dict["loss_flow"] = loss_flow * opt.lambda_flow
                    if dataset.use_flow and (float(psnr_) >= opt.flow_psnr_threshold): #and iteration > 3000
                        loss_flow, gt_fwd_flow, gt_bwd_flow, seg_map, flow_fwd_per_gs, flow_bwd_per_gs, fg_gaussian_contrib = compute_flow_loss(opt, gaussians, viewpoint_cam, render_pkg, pipe, background)
                    else: 
                        loss_flow = torch.tensor(0, device='cuda:0')
                    loss_dict["loss_flow"] = loss_flow * opt.lambda_flow * weight_scheduler.get_value(iteration)

                    # depth-normal consistency
                    # loss_surface = cos_loss(resize_image(normal, 1), resize_image(d2n, 1), thrsh=np.pi*1/10000 , weight=1)
                    loss_surface = cos_loss(normal, d2n)
                    loss_dict["loss_surface"] = (0.01 + 0.1 * min(2 * iteration / opt.iterations, 1)) * loss_surface * opt.lambda_surface

                    if mono is not None:
                        loss_monoN = cos_loss(normal, monoN, weight=gt_mask)
                    else:
                        loss_monoN = torch.tensor(0, device='cuda:0')
                    loss_dict["loss_monoN"] = (0.04 - ((iteration / opt.iterations)) * 0.03) * loss_monoN

                    # curv_n = normal2curv(normal, mask_vis)
                    # loss_curv = l1_loss(curv_n * 1, 0) #+ 1 * l1_loss(curv_d2n, 0)                
                    # total_loss += (0.005 - ((iteration / opt.iterations)) * 0.0) * loss_curv
                    
                    # encourage _trbf_scale to be larger, so Gaussians have longer lifespan
                    # e^trbf_scale == 1.414 * std_Dev
                    target_value = -1
                    clip_value = 0
                    y_clipped = torch.clamp(gaussians.get_trbfscale, max=clip_value)
                    velocities = torch.norm(gaussians.get_motion[:, 0:3], dim=-1)
                    tscale_loss_weight = 1 / (1 + velocities)
                    
                    if opt.t_scale_loss == "softplus":
                        loss_t_scale = torch.nn.functional.softplus(target_value - y_clipped.mean()) if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter else torch.tensor(0, device='cuda:0')
                    elif opt.t_scale_loss == "sigmoid":
                        loss_t_scale = torch.sigmoid(10 * (target_value - y_clipped.mean())) if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter else torch.tensor(0, device='cuda:0')
                    elif opt.t_scale_loss == "reciprocal":
                        loss_t_scale = 1 / torch.exp(0.3 * gaussians.get_trbfscale).mean() if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter else torch.tensor(0, device='cuda:0')
                    else:
                        loss_t_scale = torch.tensor(0, device='cuda:0')
                    loss_dict["loss_trbfscale"] = (loss_t_scale * tscale_loss_weight).mean() * opt.lambda_t_scale
                    total_loss = loss_dict["loss_rgb"] \
                                + loss_dict["loss_mask"] \
                                + loss_dict["loss_opac"] \
                                + loss_dict["loss_surface"] \
                                + loss_dict["loss_trbfscale"] \
                                + loss_dict["loss_monoN"] \
                                + loss_dict["loss_smooth_coeff"] \
                                + loss_dict["loss_shape_reg"] \
                                + loss_dict["loss_flow"]
                    loss_dict["total_loss"] = total_loss
                    
                # Backward pass 
                with time_loggers["backward_pass"].time_block():
                    total_loss.backward()
                    if opt.batch > 1: 
                        gaussians.cache_gradient()
                        gaussians.optimizer.zero_grad(set_to_none = True)

            iter_end.record()
            if opt.batch > 1: gaussians.set_batch_gradient(opt.batch)

                
            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss_rgb.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}, Pts={len(gaussians._xyz)}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log
                with time_loggers["training_report"].time_block():
                    test_background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
                    training_report(tb_writer, iteration, loss_dict, l1_loss, iter_start.elapsed_time(iter_end), args.test_iterations, scene, pipe, test_background, use_mask, dataset.duration)
                
                # Save
                if (iteration in args.save_iterations):
                    with time_loggers["gaussian_save"].time_block():
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)
                
                # Densification
                if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter:
                    with time_loggers["densification"].time_block():
                        # Keep track of max radii in image-space for pruning
                        # Caution: these two lines only consider the last batch!
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration % opt.knn_prune_interval == 0:
                            gaussians.knn_prune(opt.knn_prune_ratio)
                        if iteration % opt.pruning_interval == 0:
                            prune_log_info = gaussians.adaptive_prune(opt.min_opacity, scene.cameras_extent)
                            prune_report(tb_writer, prune_log_info, iteration)
                        if iteration % opt.densification_interval == 0: 
                            # gaussians.adaptive_prune(opt.min_opacity, scene.cameras_extent)  
                            gaussians.adaptive_densify(opt.densify_grad_threshold, scene.cameras_extent)
                        
                        if (iteration - 1) % opt.opacity_reset_interval == 0 and opt.opacity_lr > 0:
                            gaussians.reset_opacity(opt.opacity_reset_ratio)

                # Write progress every 1000 iterations
                if (iteration - 1) % 1000 == 0 and args.write_progress:
                    with time_loggers["write_progress"].time_block():
                        normal_wrt = normal2rgb(normal, mask_vis)
                        depth_wrt = depth2rgb(depth, mask_vis)
                        mask_wrt = gt_mask.repeat(3, 1, 1)
                        mask_vis_wrt = mask_vis.repeat(3, 1, 1)
                        opac_wrt = opac.detach().repeat(3, 1, 1)
                        if dataset.use_flow and (float(psnr_) >= opt.flow_psnr_threshold) and flow_fwd_per_gs is not None and flow_bwd_per_gs is not None:
                            gt_fwd_flow_viz, pred_fwd_flow, gt_bwd_flow_viz, pred_bwd_flow = get_flow_images(gt_fwd_flow, gt_bwd_flow, seg_map, flow_fwd_per_gs, flow_bwd_per_gs, fg_gaussian_contrib)
                            log_images = [gt_image, mask_wrt, normal_wrt * opac, gt_fwd_flow_viz, gt_bwd_flow_viz, image, opac_wrt, depth_wrt * opac, pred_fwd_flow, pred_bwd_flow]
                            progress_img = torch.cat([torch.cat(log_images[:5], dim=2), torch.cat(log_images[5:], dim=2)], dim=1) # Concatenate images in 2 rows and stack them vertically
                        # gt_bwd_flow = torch.from_numpy(flow_viz.flow_to_image(gt_bwd_flow.cpu().numpy().transpose(1, 2, 0)).transpose(2, 0, 1)).float().to('cuda:0') / 255.0
                        else:
                            progress_img = torch.cat([gt_image, image, normal_wrt * opac, depth_wrt * opac, mask_wrt, mask_vis_wrt, opac_wrt], 2)
                        
                        progress_path = f'{dataset.model_path}/progress'
                        os.makedirs(progress_path, exist_ok=True)
                        save_image(progress_img.cpu(), f'{progress_path}/it{iteration-1}.png')
                
                # Optimizer step
                if iteration < opt.iterations:
                    with time_loggers["optimizer_step"].time_block():
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)

                # Saving checkpoint
                if (iteration in args.checkpoint_iterations):
                    with time_loggers["checkpoint_save"].time_block():
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
                # Log timing stats
                if iteration % 1000 == 0:
                    for name, logger in time_loggers.items():
                        avg_time = logger.get_average_duration(clear_after=True, unit="milliseconds")
                        if avg_time is not None:
                            tb_writer.add_scalar(f"Time/{name}", avg_time, iteration)
                            logger.reset()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/test516", f"{args.source_path.split('/')[-1]}_{unique_str[0:10]}")
        
    # Set up output folder
    start_frame = int(re.search(r'colmap_(\d+)$', args.source_path).group(1))
    output_path = os.path.join(args.model_path, f"{start_frame}to{start_frame+args.duration-1}")
    setattr(args, 'model_path', output_path)
    setattr(args, 'start_frame', start_frame)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    
    args_dict = vars(args)
    with open(os.path.join(args.model_path, "cfg_args.json"), 'w') as cfg_log_f:
        json.dump(args_dict, cfg_log_f, indent=4)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss_dict, l1_loss, elapsed, testing_iterations, scene : Scene, pipe, bg, use_mask, duration):
    if tb_writer:
        for loss_name, loss_value in loss_dict.items():
            tb_writer.add_scalar(f'train_loss_patches/{loss_name}', loss_value.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            #   {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
                              )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                masked_psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(render(viewpoint, scene.gaussians, pipe, bg)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.get_gtImage(bg, with_mask=use_mask), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_image(config['name'] + "_view_{}/render".format(viewpoint.image_name), image, global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_image(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image, global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    masked_psnr_test += masked_psnr(image, gt_image)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])  
                masked_psnr_test /= len(config['cameras'])        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/masked_psnr', masked_psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("histograms/opacity", scene.gaussians.get_base_opacity, iteration)
            tb_writer.add_histogram("histograms/trbfcenter", scene.gaussians.get_trbfcenter, iteration)
            tb_writer.add_histogram("histograms/trbfscale", scene.gaussians.get_trbfscale, iteration)
            
            if scene.gaussians.motion_degree >= 3: 
                tb_writer.add_histogram('motion/b_x0', scene.gaussians.get_motion[:, 0], iteration)
                tb_writer.add_histogram('motion/b_y0', scene.gaussians.get_motion[:, 1], iteration)
                tb_writer.add_histogram('motion/b_z0', scene.gaussians.get_motion[:, 2], iteration)
                tb_writer.add_histogram('motion/b_x1', scene.gaussians.get_motion[:, 3], iteration)
                tb_writer.add_histogram('motion/b_y1', scene.gaussians.get_motion[:, 4], iteration)
                tb_writer.add_histogram('motion/b_z1', scene.gaussians.get_motion[:, 5], iteration)
                tb_writer.add_histogram('motion/b_x2', scene.gaussians.get_motion[:, 6], iteration)
                tb_writer.add_histogram('motion/b_y2', scene.gaussians.get_motion[:, 7], iteration)
                tb_writer.add_histogram('motion/b_z2', scene.gaussians.get_motion[:, 8], iteration)
                
            # show opac distribution of the last iter. will overwrite previous iters
            sum_ratio = 0
            for timestep in range(0, duration):
                t = timestep/duration
                opacity = scene.gaussians.get_full_opacity(t)
                ratio_points_above_threshold = (opacity > 0.5).sum().item() / opacity.shape[0]
                avg_opacity = opacity.mean().item()
                tb_writer.add_scalar('opacity/above_thresh_ratio_final', ratio_points_above_threshold, global_step=timestep)
                tb_writer.add_scalar('opacity/average_opacity', avg_opacity, global_step=timestep)
                sum_ratio += ratio_points_above_threshold
            avg_ratio_over_frames = sum_ratio / duration
            tb_writer.add_scalar('opacity/above_thresh_ratio_avg', avg_ratio_over_frames, iteration)
            
            # show position gradient norm histogram
            grad_pos = scene.gaussians.xyz_gradient_accum / scene.gaussians.denom
            grad_pos[grad_pos.isnan()] = 0.0
            tb_writer.add_histogram("histograms/grad_norms", torch.norm(grad_pos, dim=-1), iteration)
            
            # log gradient scales
            scale_min = scene.gaussians.get_scaling[:,:2].min().mean().item()
            scale_max = scene.gaussians.get_scaling[:,:2].max().mean().item()
            axis_3 = scene.gaussians.get_scaling[:,2].mean().item()
            tb_writer.add_scalar('scale/min_avg', scale_min, iteration)
            tb_writer.add_scalar('scale/max_avg', scale_max, iteration)
            tb_writer.add_scalar('scale/3rd_axis_avg', axis_3, iteration)
            tb_writer.add_scalar('scale/trbfscale_avg', scene.gaussians.get_trbfscale.mean().item(), iteration)
            
        torch.cuda.empty_cache()

def prune_report(tb_writer, prune_log_info, iteration):
    if tb_writer:
        for prune_log_name, prune_log_value in prune_log_info.items():
            tb_writer.add_scalar(f'pruning/{prune_log_name}', prune_log_value.item(), iteration)
        
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, required=False, default=None, help="Path to the configuration YAML file")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--write_progress", action="store_true", default=True)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint", type=str, default = None)
    
    args = parser.parse_args()
    if os.path.exists(args.config) and args.config != "None":
        config = load_config_file(args.config)
        args = get_combined_args(args, config)
    
    args.save_iterations.append(args.iterations)
    args.test_iterations = list(range(0, 100000, 1000))
    print("Optimizing " + args.model_path)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    setgtisint8(pp.extract(args).gtisint8)
    training(args)

    # All done
    print("\nTraining complete.")
