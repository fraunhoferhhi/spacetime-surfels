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

from scene.cameras import Camera
import numpy as np
import torch
from utils.general_utils import PILtoTorch, TorchImageResize
from utils.graphics_utils import fov2focal
from utils.image_utils import resize_image
from tqdm import tqdm

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    # orig_w, orig_h = cam_info.image.size
    orig_h, orig_w = cam_info.image.shape[1], cam_info.image.shape[2]

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    # cam_info.image.close()
    resized_image_rgb = TorchImageResize(cam_info.image, resolution)

    resized_mono = None if cam_info.mono is None else resize_image(cam_info.mono, [resolution[1], resolution[0]])
    resized_flow_fwd = None if cam_info.flow_fwd is None else resize_image(cam_info.flow_fwd, [resolution[1], resolution[0]])
    resized_flow_bwd = None if cam_info.flow_bwd is None else resize_image(cam_info.flow_bwd, [resolution[1], resolution[0]])

    gt_image = resized_image_rgb[:3, ...]
    
    if cam_info.mask is not None: # from "mask" folder
        loaded_mask = resize_image(cam_info.mask, [resolution[1], resolution[0]])
    elif resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]    
    else: assert False, "no mask input"

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, K=cam_info.K,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, prcppoint=cam_info.prcppoint,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  mono=resized_mono, flow_fwd=resized_flow_fwd, flow_bwd=resized_flow_bwd, timestamp=cam_info.timestamp)

# scene_scale == cameras_extent == radius == max. dist. from cam center to cams
def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos, unit="cameras", leave=False)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width),
        'prcp': camera.prcppoint.tolist(),
        'timestamp': camera.timestamp
    }
    return camera_entry
