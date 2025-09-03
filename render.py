import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from os.path import dirname, join
from gaussian_renderer import render
from utils.general_utils import safe_state, poisson_mesh, setgtisint8
from utils.image_utils import psnr, depth2rgb, normal2rgb, resample_points, grid_prune, error_map
from argparse import ArgumentParser
from torchvision.utils import save_image
from arguments import ModelParams, PipelineParams, get_combined_args, load_config_file
from gaussian_renderer import GaussianModel
import ffmpeg
import copy
from scipy.ndimage import label
# from submodules.RAFT.utils import flow_viz

def render_dynamic(dataset, use_mask, name, iteration, cameras, gaussians, pipeline, background, args):
    unique_timestamps = sorted(set(cam.timestamp for cam in cameras))

    render_dir = join(dataset.model_path, "..", "render")
    frames_dir = join(render_dir, name, f"it{iteration}")
    mesh_dir = join(dataset.model_path, "..", "meshes_pymeshlab" if args.use_pymeshlab else "meshes_o3d", f"poisson_it{iteration}_depth{args.poisson_depth}")
    makedirs(mesh_dir, exist_ok=True)
    
    for t_idx, timestamp in enumerate(tqdm(unique_timestamps, desc="Timestamp progress")):
        f_idx = t_idx + gaussians.start_frame
        
        if args.write_mesh and name == 'train':
            grid_dim = 512 if args.poisson_depth <=9 else 1024
            occ_grid, grid_shift, grid_scale, grid_dim = gaussians.to_occ_grid(0.0, grid_dim, None, timestamp, args.opac_thrsh_4_occ_grid)

        resampled = []
        psnr_all = []
                
        views = [cam for cam in cameras if cam.timestamp == timestamp]
        
        for v_idx, view in enumerate(tqdm(views, desc="Rendering progress", leave=False)):
            view_dir = join(frames_dir, f"v{view.image_name}")  
            makedirs(view_dir, exist_ok=True)

            render_pkg = render(view, gaussians, pipeline, background)
            
            image, normal, depth, opac = \
                render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"]

            gt_mask = view.get_gtMask(use_mask)
            gt_image = view.get_gtImage(background, use_mask).cuda()
            psnr_all.append(psnr((gt_image).to(torch.float64), (image).to(torch.float64)).mean().cpu().numpy())
            mask_vis = (opac.detach() > 0.5) #1e-1)
            depth_range = [0, 20]
            mask_clip = (depth > depth_range[0]) * (depth < depth_range[1])
            normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis
            # gt_flow_fwd, _ = view.get_gtFlow() if dataset.use_flow else (None, None)

            if args.write_mesh and name == 'train':
                # error_map_ = error_map((gt_image).to(torch.float64), (image).to(torch.float64))
                # mask_rendered_error = (error_map_ > 0) & (error_map_ < 0.001)
                # error_path = os.path.join(view_dir, f'mask_rendered_error_f{f_idx:04d}.png')
                # save_image(mask_rendered_error.detach().cpu().float(), error_path)

                # unproject filtered depth map to 3D points in world space
                # [H, W, 9(xyz_in_world, normals, rgb)]
                pts = resample_points(view, depth, normal, image, mask_vis * gt_mask * mask_clip) # * mask_rendered_error)
                grid_mask = grid_prune(occ_grid, grid_shift, grid_scale, grid_dim, pts[..., :3], thrsh=args.occ_thrsh)
                pts = pts[grid_mask]
                resampled.append(pts.cpu())

            if args.write_img and name == 'test':
                mask_vis = (opac.detach() > 0.5)
                
                # Create a mask for the largest component
                # np_mask = mask_vis.cpu().numpy()
                # labeled_array, num_features = label(np_mask)
                # largest_component = np.argmax(np.bincount(labeled_array.flatten())[1:]) + 1
                # largest_component_mask = labeled_array == largest_component
                # mask_vis = torch.tensor(largest_component_mask, dtype=torch.bool).to(mask_vis.device)
    
                normal_wrt = normal2rgb(normal, mask_vis)
                depth_wrt = depth2rgb(depth, mask_vis)
                mask_wrt = gt_mask.repeat(3, 1, 1)
                mask_vis_wrt = mask_vis.repeat(3, 1, 1)
                opac_wrt = opac.detach().repeat(3, 1, 1)

                # if dataset.use_flow:
                #     flow_wrt = torch.from_numpy(flow_viz.flow_to_image(flow.detach().cpu().numpy().transpose(1, 2, 0)).transpose(2, 0, 1)).float().to('cuda:0') / 255.0	
                #     gt_flow_wrt = torch.from_numpy(flow_viz.flow_to_image(gt_flow_fwd.cpu().numpy().transpose(1, 2, 0)).transpose(2, 0, 1)).float().to('cuda:0') / 255.0
                #     log_images = [gt_image, mask_wrt, normal_wrt * opac, gt_flow_wrt, image, opac_wrt, depth_wrt * opac, flow_wrt]
                #     img_wrt = torch.cat([torch.cat(log_images[:4], dim=2), torch.cat(log_images[4:], dim=2)], dim=1) # Concatenate images in 2 rows and stack them vertically
                # else:
                img_wrt = torch.cat([gt_image, image, normal_wrt * opac, depth_wrt * opac, mask_wrt, mask_vis_wrt, opac_wrt], 2)

                info_path = os.path.join(view_dir, f'info_f{f_idx:04d}.png')
                save_image(img_wrt.detach().cpu(), info_path)
                gt_path = os.path.join(view_dir, f'gt_f{f_idx:04d}.png')
                masked_gt = torch.cat((gt_image, gt_mask), dim=0)
                save_image(masked_gt.detach().cpu(), gt_path)
                masked_pred = torch.cat((image, mask_vis), dim=0)
                pred_path = os.path.join(view_dir, f'pred_f{f_idx:04d}.png')
                save_image(masked_pred.detach().cpu(), pred_path)
                
                # normal_path = os.path.join(view_dir, f'normal_f{f_idx:04d}.png')
                # masked_normal = torch.cat((normal_wrt, mask_vis), dim=0)
                # save_image(masked_normal.detach().cpu(), normal_path)
                # depth_path = os.path.join(view_dir, f'depth_f{f_idx:04d}.png')
                # masked_depth = torch.cat((depth_wrt, mask_vis), dim=0)

        eval_result_path = join(dataset.model_path, "eval_result.txt")
        with open(eval_result_path, 'a') as f:
            f.write(f'PSNR_{name}_{timestamp}: {np.mean(psnr_all)}\n')

        if args.write_mesh and name == 'train':
            resampled = torch.cat(resampled, 0)
            mesh_path = f'{mesh_dir}/f{f_idx:04d}'            
            poisson_mesh(mesh_path, resampled[:, :3], resampled[:, 3:6], resampled[:, 6:], args.poisson_depth, args.use_pymeshlab)

    return frames_dir
    

def render_sets(dataset : ModelParams, pipeline : PipelineParams, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset)

        scales = [1]
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, resolution_scales=scales)

        bg_color = [1,1,1] if not args.write_img else [0, 0, 0] # if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not args.skip_test:
            print("Rendering test set")
            render_dynamic(dataset, True, "test", scene.loaded_iter, scene.getTestCameras(scales[0]), gaussians, pipeline, background, args)

        if not args.skip_train:
            print("Rendering train set")
            render_dynamic(dataset, True, "train", scene.loaded_iter, scene.getTrainCameras(scales[0]), gaussians, pipeline, background, args)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--parent_path', type=str, required=True, help="Parent directory containing model folders")
    parser.add_argument('--config', type=str, required=False, default=None, help="Path to the configuration YAML file")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--write_img", action="store_true")
    parser.add_argument("--write_mesh", action="store_true")
    parser.add_argument("--write_video", action="store_true")
    parser.add_argument("--poisson_depth", default=9, type=int)
    parser.add_argument("--use_pymeshlab", action="store_true")
    parser.add_argument("--occ_thrsh", default=1., type=float)
    parser.add_argument("--opac_thrsh_4_occ_grid", default=0., type=float)
    
    original_args = parser.parse_args()
    
    for model_dir in sorted(os.listdir(original_args.parent_path)):
        args = copy.deepcopy(original_args)
        model_path = os.path.join(args.parent_path, model_dir)
        args.model_path = model_path
        cfg_path = os.path.join(model_path, "cfg_args.json")
        
        if not os.path.isdir(model_path) or not os.path.exists(cfg_path):
            continue
       
        config = load_config_file(cfg_path)
        args = get_combined_args(args, config)
        
        print(f"Rendering {os.path.relpath(model_path, args.parent_path)}")
        safe_state(args.quiet)
        setgtisint8(pipeline.extract(args).gtisint8)
        
        frames_dir = render_sets(model.extract(args), pipeline.extract(args), args)
        
        
    # if args.write_video:
    #     breakpoint()
    #     for view in tqdm(os.listdir(frames_dir), desc="Creating videos"):
    #         video_path = join(frames_dir, f"{view}.mp4")
    #         try:
    #             (
    #                 ffmpeg
    #                 .input(join(frames_dir, view, 'info_f%04d.png'), framerate=10, start_number=gaussians.start_frame)  # Adjust framerate as needed
    #                 .output(video_path)
    #                 .global_args('-loglevel', 'info')  # Use 'info' to see more details in stderr
    #                 .global_args('-y')
    #                 .run(capture_stdout=True, capture_stderr=True)  # Capture stdout and stderr
    #             )
    #         except ffmpeg.Error as e:
    #             print(f"Error creating video for {view}")
    #             print(e.stderr.decode('utf-8'))  # Decode and print stderr to see details of the error