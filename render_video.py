import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state, poisson_mesh, setgtisint8
from utils.image_utils import psnr, depth2rgb, normal2rgb, depth2normal, resample_points, grid_prune, depth2viewDir, img2video
from argparse import ArgumentParser
from torchvision.utils import save_image
from arguments import ModelParams, PipelineParams, get_combined_args, load_config_file
from gaussian_renderer import GaussianModel
from utils.render_utils import generate_path, create_videos, get_largest_mesh_folder, render_mesh
import sys
import json
import json5
import time
import re
import copy
from argparse import Namespace
from scipy.ndimage import label
# from submodules.RAFT.utils import flow_viz

@torch.no_grad()
def render_maps(dataset: ModelParams, gaussians: GaussianModel, pipeline : PipelineParams, rendered_dict: dict, cam):
    # gaussians = GaussianModel(dataset)
    # scales = [1]
    # scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False, resolution_scales=scales)
    bg_color = [1,1,1] # if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    render_pkg = render(cam, gaussians, pipeline, background)

    image, normal, depth, opac = \
                render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"]
    
    mask_vis = (opac.detach() > 0.5) #1e-1) #1e-5

    # Create a mask for the largest component
    np_mask = mask_vis.cpu().numpy()
    labeled_array, num_features = label(np_mask)
    largest_component = np.argmax(np.bincount(labeled_array.flatten())[1:]) + 1
    largest_component_mask = labeled_array == largest_component
    mask_vis = torch.tensor(largest_component_mask, dtype=torch.bool).to(mask_vis.device)

    normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis
    normal_wrt = normal2rgb(normal, mask_vis, background)
    depth_wrt = depth2rgb(depth, mask_vis, background)

    # rendered_dict["rgbs"].append((image * mask_vis).permute(1,2,0).cpu().numpy())
    # Set the background to white (1.0) where the mask is False
    image_rgb = image.permute(1, 2, 0).cpu().numpy()
    mask_vis = mask_vis.permute(1, 2, 0).cpu().numpy()
    white_bg_image = image_rgb * mask_vis + (1 - mask_vis)
    # flow_wrt = flow_viz.flow_to_image(flow.detach().cpu().numpy().transpose(1, 2, 0)) / 255.0	
    rendered_dict["rgbs"].append(white_bg_image)
    rendered_dict["normals"].append(normal_wrt.permute(1,2,0).cpu().numpy())
    rendered_dict["depths"].append(depth_wrt.permute(1,2,0).cpu().numpy())
    # rendered_dict["flows"].append(flow_wrt)

def reset_timestamps(cam_traj):
    for i, cam in enumerate(cam_traj):
        cam.timestamp = i
    return cam_traj

def create_camera_trajectory(first_frame_cams, args, total_frames, frames):
    n_frames = total_frames
    trajectories = []
    current_frames = frames.copy()
    
    # Base trajectory
    cam_traj = generate_path(first_frame_cams, args.duration, n_frames, 
                           args.intrinsics_cam_idx, args.extrinsics_cam_idx, args.rotate)
    
    if args.rotation_offset:
        original_timestamps = [cam.timestamp for cam in cam_traj]
        cam_traj = cam_traj[args.rotation_offset:] + cam_traj[:args.rotation_offset]
        for i, cam in enumerate(cam_traj):
            cam.timestamp = original_timestamps[i]
    
    if args.stop_at:
        # Create intermediate circle
        intermediate_traj = copy.deepcopy(cam_traj)
        stop_cam = cam_traj[args.stop_at]
        
        for cam in intermediate_traj:
            cam.timestamp = stop_cam.timestamp
            
        intermediate_traj = (intermediate_traj[args.stop_at:] + 
                           intermediate_traj[:args.stop_at])
        
        cam_traj = (cam_traj[:args.stop_at] + 
                   intermediate_traj + 
                   cam_traj[args.stop_at:])
                   
        current_frames = (frames[:args.stop_at] + 
                         [frames[args.stop_at]] * n_frames + 
                         frames[args.stop_at:])
                         
        n_frames *= 2
        
    return cam_traj, current_frames, n_frames

def render_sequence(dataset, gaussians, pipeline_args, start_frame, frames, cam_traj, args, rendered_dict):
    for cam_idx, cam in enumerate(tqdm(cam_traj, desc="Rendering frames")):
        cam.update_intrinsics(args.scale_f_len, args.rendered_h, 
                            args.rendered_w, args.d_cy)
        
        frame_idx = start_frame + frames[cam_idx]
        if args.neus2_mesh_dir is None:
            # Render maps
            render_maps(dataset, gaussians, pipeline_args, rendered_dict, cam)
            mesh_dir = get_largest_mesh_folder(os.path.join(args.parent_path, "meshes_o3d"))
            mesh_path = os.path.join(mesh_dir, f"f{frame_idx:06d}_cleaned.ply")
        else:
            mesh_path = os.path.join(args.neus2_mesh_dir, f"frame_{frame_idx:06d}.obj")

        # Render mesh
        render_mesh(mesh_path, cam, rendered_dict)
        
    return rendered_dict

def get_start_frame(dir_name):
    try:
        return int(dir_name.split('to')[0])
    except (ValueError, IndexError):
        return float('inf')  # Put non-matching directories at the end

def camera_to_dict(cameras):
    """
    Extract R and T matrices from a list of Camera objects and store in a dictionary
    
    Args:
        cameras: A list of Camera() objects
        
    Returns:
        A dictionary with camera indices as keys and R, T values as values
    """
    camera_dict = {}
    
    for i, camera in enumerate(cameras):
        # Extract R (rotation matrix) and T (translation vector)
        R = camera.R  # rotation matrix (c2w)
        T = -R @ camera.T  # translation vector in world coordinates
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(R, np.ndarray):
            R = R.tolist()
        if isinstance(T, np.ndarray):
            T = T.tolist()
        
        # Store in dictionary with camera index as key
        camera_dict[f"camera_{i}"] = {
            "R": R,
            "T": T
        }
    
    return camera_dict

if __name__ == "__main__":
    # note that this script requires X11 forwarding if you are using ssh. Alternatively, one can run this script on the server machine via remote desktop
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--parent_path", type=str, required=True, help="Parent directory containing model folders")
    parser.add_argument('--config', type=str, required=False, default=None, help="Path to the configuration YAML file")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--img", action="store_true")
    parser.add_argument("--depth", default=10, type=int)
    parser.add_argument("--read_config", action='store_true', default=False)
    parser.add_argument("--config_path", type=str, default = None)
    parser.add_argument("--fps", default=25, type=int, help="fps of the output video")
    parser.add_argument("--intrinsics_cam_idx", default=13, type=int, help="the cam idx of training cams which is used for the intrinsics, h, w of the rendering cam.\
                            NHR: 13")
    parser.add_argument("--extrinsics_cam_idx", default=None, type=int, help="the cam idx of training cams which is used for the pose of the rendering cam")
    parser.add_argument("--rotate", default=False, action="store_true", help="rotate the camera around the object")
    parser.add_argument("--rotation_offset", default=None, type=int, help="offset of the created circle of cam paths")
    parser.add_argument("--stop_at", default=None, type=int, help="stop at this frame and rotate one circle")
    parser.add_argument("--scale_f_len", type=float, default = None)
    parser.add_argument("--d_cy", type=float, default = None)
    parser.add_argument("--rendered_h", type=int, default = None)
    parser.add_argument("--rendered_w", type=int, default = None)
    parser.add_argument("--start_frame", type=int, default = 0)
    parser.add_argument("--n_frames", type=int, default = 150)
    parser.add_argument("--neus2_mesh_dir", type=str, default = None)

    script_args = parser.parse_args()
    # Sort directories based on their starting frame number
    all_model_dirs = [d for d in os.listdir(script_args.parent_path) 
                 if 'to' in d and d.split('to')[0].isdigit()]
    all_model_dirs = sorted(all_model_dirs, key=get_start_frame)

    # First, generate the complete camera trajectory
    first_model_dir = all_model_dirs[0]
    args = copy.deepcopy(script_args)
    model_path = os.path.join(args.parent_path, first_model_dir)
    args.model_path = model_path
    cfg_path = os.path.join(model_path, "cfg_args.json")
    config = load_config_file(cfg_path)
    args = get_combined_args(args, config)

    if args.extrinsics_cam_idx:
        args.intrinsics_cam_idx = args.extrinsics_cam_idx
        args.fps = 10

    # Load first scene to get camera information
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, 
                shuffle=False, resolution_scales=[1], loader="video")
    first_frame_cams = sorted(scene.getTrainCameras(scale=1) + 
                            scene.getTestCameras(scale=1), 
                            key=lambda x: x.image_name)

    # Generate full trajectory for all frames
    total_frames = args.n_frames
    full_frames = list(range(total_frames))
    full_cam_traj, full_frames, _ = create_camera_trajectory(
        first_frame_cams, args, total_frames, full_frames)
    
    # # Save the camera trajectory to a JSON file (for visualization)
    # cam_dict = camera_to_dict(full_cam_traj)
    # with open('cam_path.json', 'w') as f:
    #     json.dump(cam_dict, f, indent=4)

    # Now render each segment with its corresponding gaussians
    rendered_dict = {
        "rgbs": [],
        "depths": [],
        "normals": [],
        # "flows": [],
        "meshes": []
    }

    prev_n_frames = 0
    for i, model_dir in enumerate(all_model_dirs):
        start_frame = int(model_dir.split('to')[0])  # Get actual start frame from directory name
        end_frame = int(model_dir.split('to')[1])    # Get actual end frame from directory name
        if args.stop_at:
            if args.stop_at > start_frame and args.stop_at < end_frame:
                n_segment_frames = end_frame - start_frame + 1 + args.n_frames
            else: 
                n_segment_frames = end_frame - start_frame + 1
                
        else:
            n_segment_frames = end_frame - start_frame + 1       # Calculate number of frames for this segment
        # Extract the relevant portion of the camera trajectory
        
        segment_frames = full_frames[prev_n_frames:prev_n_frames+n_segment_frames]
        segment_cam_traj = full_cam_traj[prev_n_frames:prev_n_frames+n_segment_frames]
        prev_n_frames += n_segment_frames
        # Initialize
        args = copy.deepcopy(script_args)
        model_path = os.path.join(args.parent_path, model_dir)
        args.model_path = model_path
        cfg_path = os.path.join(model_path, "cfg_args.json")
        config = load_config_file(cfg_path)
        args = get_combined_args(args, config)
        safe_state(args.quiet)
            
        # Load scene for this segment
        dataset = model.extract(args)
        setgtisint8(pipeline.extract(args).gtisint8)
        gaussians = GaussianModel(dataset)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, 
                    shuffle=False, resolution_scales=[1], loader="video")
        
        # Render this segment
        render_sequence(dataset, gaussians, 
                        pipeline.extract(args),
                        start_frame,
                        segment_frames,
                        segment_cam_traj, args,
                        rendered_dict)
    
    video_dir = os.path.join(args.parent_path, 'video')
    if args.neus2_mesh_dir:
        video_dir = os.path.join(args.neus2_mesh_dir, '../rotate_video')
    os.makedirs(video_dir, exist_ok=True)
    h = args.rendered_h if args.rendered_h else full_cam_traj[0].image_height
    w = args.rendered_w if args.rendered_w else full_cam_traj[0].image_width
    # h,w must be a multiple of 2
    create_videos(rendered_dict, video_dir, (h,w), args.fps)
