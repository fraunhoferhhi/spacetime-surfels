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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from torchvision.utils import save_image
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.image_utils import resize_image
from glob import glob
import imageio
import skimage
import torchvision.io as io
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    prcppoint: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    flow_fwd: np.array
    flow_bwd: np.array
    mono: np.array
    timestamp: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    start_frame: int

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, mono_normal, use_flow, timestamp):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            prcppoint = np.array([intr.params[1] / width, intr.params[2] / height])
            K = np.array([
                [focal_length_x, 0, intr.params[1]],
                [0, focal_length_x, intr.params[2]],
                [0, 0, 1]
            ], dtype=np.float32)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            prcppoint = np.array([intr.params[2] / width, intr.params[3] / height])
            K = np.array([
                [focal_length_x, 0, intr.params[2]],
                [0, focal_length_y, intr.params[3]],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
        
        # image = Image.open(image_path) # numpy array, float32, [0, 1]
        image = io.read_image(image_path) # torch tensor, uint8, [0, 255]

        monoN_path = image_path.replace("images", "normal").replace(".png", "_normal.npy")
        monoD_path = image_path.replace("images", "depth").replace(".png", "_depth.npy")
        mask_path = image_path.replace("images", "masks")
        fwd_flow_path = image_path.replace("images", "flow/flow_fwd").replace(".png", ".npy")
        bwd_flow_path = image_path.replace("images", "flow/flow_bwd").replace(".png", ".npy")

        # monococular priors
        if mono_normal:
            try:
                monoN = read_monoData(monoN_path)
                try:
                    monoD = read_monoData(monoD_path)
                except FileNotFoundError:
                    monoD = np.zeros_like(monoN[:1])
                mono = np.concatenate([monoN, monoD], 0)
            except FileNotFoundError:
                mono = None
        else: 
            mono = None
            
        
        if use_flow:
            try:
                flow_fwd = read_monoData(fwd_flow_path).transpose(2, 0, 1)
            except FileNotFoundError:
                flow_fwd = None
            try:
                flow_bwd = read_monoData(bwd_flow_path).transpose(2, 0, 1)
            except FileNotFoundError:
                flow_bwd = None
        else:
            flow_fwd = None
            flow_bwd = None

        # mask
        # try:
        mask = load_mask(mask_path)[None]
        # except FileNotFoundError:
        #     mask = np.ones([1, image.size[1], image.size[0]]).astype(np.float32)
        
        # # binary mask to avoid floating problem when multiplied with depth in render.py
        # mask = (mask > 0.5).astype(np.float32)

        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=FovY, FovX=FovX, prcppoint=prcppoint, image=image,
                        image_path=image_path, image_name=image_name, width=width, height=height, mask=mask, 
                        mono=mono, flow_fwd=flow_fwd, flow_bwd=flow_bwd, timestamp=timestamp)
        cam_infos.append(cam_info)
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    times = np.vstack([vertices['t']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals, times=times)

def storePly(path, xyzt, rgb, normal=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('t','f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    xyz = xyzt[:, :3]
    normals = np.zeros_like(xyz) if normal is None else normal

    elements = np.empty(xyzt.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzt, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, mono_normal, use_flow, llffhold=8, duration=50, test_views=[0], random_init=False):
    colmap, start_frame = os.path.basename(path).split("_") # ex: colmap, 0 
    assert start_frame.isdigit(), "Colmap folder name must be colmap_<startime>!"
    start_frame = int(start_frame)

    cam_infos_unsorted = []
    for t in range(start_frame, start_frame+duration):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading colmap folders {}/{}".format(t-start_frame+1, duration))
        sys.stdout.flush()
        new_path = os.path.join(os.path.dirname(path), f"{colmap}_{t}") # ex: colmap_0 -> colmap_1
        try:
            cameras_extrinsic_file = os.path.join(new_path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(new_path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(new_path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(new_path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        # ordered-time-major, unordered cams/views
        cam_infos_unsorted += readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, \
                                                images_folder=os.path.join(new_path, "images"), mono_normal=mono_normal, \
                                                use_flow=use_flow, timestamp=(t-start_frame)/duration)
    sys.stdout.write('\n')
    
    # orderd-view-major, ordered times; image_name is view id
    # [..., colmap_48/images/59.png, colmap_49/images/59.png]
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    cam_infos = sorted(cam_infos_unsorted, key = lambda x : x.image_name)

    if eval:
        test_cam_infos = []        
        if test_views:
            # Collect indices for test views
            test_indices = set()
            for view in test_views:
                start_idx = view * duration
                end_idx = (view + 1) * duration
                test_cam_infos += cam_infos[start_idx:end_idx]
                test_indices.update(range(start_idx, end_idx))

            # Remove test view indices from training views
            train_cam_infos = [info for idx, info in enumerate(cam_infos) if idx not in test_indices]
        else: 
            train_cam_infos = cam_infos[duration:]
            test_cam_infos = cam_infos[:duration]
        uniquecheck = []
        for cam_info in test_cam_infos:
            if cam_info.image_name not in uniquecheck:
                uniquecheck.append(cam_info.image_name)
        sanitycheck = []
        for cam_info in train_cam_infos:
            if cam_info.image_name not in sanitycheck:
                sanitycheck.append(cam_info.image_name)
        for testname in uniquecheck:
            assert testname not in sanitycheck
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos[:2] #dummy

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    totalply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")

    if not os.path.exists(totalply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        totalxyz = []
        totalrgb = []
        totaltime = []
        for i in range(start_frame, start_frame + duration):
            thisbin_path = os.path.join(path, "sparse/0/points3D.bin").replace("colmap_"+ str(start_frame), "colmap_" + str(i), 1)
            xyz, rgb, _ = read_points3D_binary(thisbin_path)
            totalxyz.append(xyz)
            totalrgb.append(rgb)
            totaltime.append(np.ones((xyz.shape[0], 1)) * (i-start_frame) / duration)
        xyz = np.concatenate(totalxyz, axis=0)
        rgb = np.concatenate(totalrgb, axis=0)
        totaltime = np.concatenate(totaltime, axis=0)
        assert xyz.shape[0] == rgb.shape[0]  
        xyzt = np.concatenate( (xyz, totaltime), axis=1)     
        storePly(totalply_path, xyzt, rgb)
    try:
        pcd = fetchPly(totalply_path)
    except:
        pcd = None
        
    if random_init:
        print("Random initialization")
        # generate random points in the bounding box of totalply
        bbox_min = np.min(pcd.points, axis=0)  # Minimum x, y, z values
        bbox_max = np.max(pcd.points, axis=0)  # Maximum x, y, z values
        num_pts = 500_000
        random_points = np.random.rand(num_pts, 3)  # Generate points in the range [0, 1]
        random_points = bbox_min + random_points * (bbox_max - bbox_min)
        shs = np.random.random((num_pts, 3)) / 255.0
        times = np.random.rand(num_pts, 1)
        pcd = BasicPointCloud(points=random_points, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)), times=times)
        

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=totalply_path,
                           start_frame=start_frame)
    return scene_info


def load_mask(path):
    alpha = imageio.imread(path, pilmode='F')
    alpha = skimage.img_as_float32(alpha) / 255
    return alpha

def read_monoData(path):
    mono = np.load(path)
    if len(mono.shape) == 4:
        mono = mono[0]
    elif len(mono.shape) == 2:
        mono = mono[None]
    return mono

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo
}
