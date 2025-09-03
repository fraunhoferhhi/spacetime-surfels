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
from utils.graphics_utils import fov2focal, focal2fov
from torch.utils.cpp_extension import load
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.nn.functional as F
import glob
import cv2
from tqdm import tqdm

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def masked_psnr(img1, img2):
    # img2 is gt, where bg is 0
    mask = (img2 > 0).float()    
    # Apply mask to images
    masked_diff = (img1 - img2) * mask    
    # Compute MSE only for the masked regions
    mse = (masked_diff ** 2).sum() / mask.sum()    
    # Handle edge case where mask.sum() might be zero
    if mask.sum() == 0:
        return float('inf')    
    # Compute PSNR
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))    
    return psnr_value.item()

def error_map(img1, img2):
    mse = (((img1 - img2)) ** 2).mean(0, keepdim=False) # mean of non-zero values is 0.0042
    return mse

def depth2rgb(depth, mask, background=None):
    sort_d = torch.sort(depth[mask.to(torch.bool)])[0]
    min_d = sort_d[len(sort_d) // 100 * 5]
    max_d = sort_d[len(sort_d) // 100 * 95]
    depth = (depth - min_d) / (max_d - min_d) * 0.9 + 0.1
    
    viridis = ListedColormap(plt.cm.viridis(np.linspace(0, 1, 256)))
    depth_draw = viridis(depth.detach().cpu().numpy()[0])[..., :3]
    depth_draw = torch.from_numpy(depth_draw).to(depth.device).permute([2, 0, 1]) * mask

    if background is not None: 
        # depth_draw += background[0] * (~mask)
        depth_draw += background[0] * (1-mask.float())
    return depth_draw

def normal2rgb(normal, mask, background=None):
    normal_draw = torch.cat([normal[:1], -normal[1:2], -normal[2:]])
    normal_draw = (normal_draw * 0.5 + 0.5) * mask
    
    if background is not None: 
        # normal_draw += background[0] * (~mask)
        normal_draw += background[0] * (1-mask.float())
    return normal_draw

def depth2normal(depth, mask, camera):
    # conver to camera position
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]), indexing='ij')
    # print(h)
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)
    
    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2,2])
    Kinv = torch.inverse(K).to(device)
    # print(p.shape, Kinv.shape)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    # padded = mod.contour_padding(camPos.contiguous(), mask.contiguous(), torch.zeros_like(camPos), filter_size // 2)
    # camPos = camPos + padded
    p = torch.nn.functional.pad(camPos[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    mask = torch.nn.functional.pad(mask[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)
    

    p_c = (p[:, 1:-1, 1:-1, :]      ) * mask[:, 1:-1, 1:-1, :]
    p_u = (p[:,  :-2, 1:-1, :] - p_c) * mask[:,  :-2, 1:-1, :]
    p_l = (p[:, 1:-1,  :-2, :] - p_c) * mask[:, 1:-1,  :-2, :]
    p_b = (p[:, 2:  , 1:-1, :] - p_c) * mask[:, 2:  , 1:-1, :]
    p_r = (p[:, 1:-1, 2:  , :] - p_c) * mask[:, 1:-1, 2:  , :]

    n_ul = torch.linalg.cross(p_u, p_l)
    n_ur = torch.linalg.cross(p_r, p_u)
    n_br = torch.linalg.cross(p_b, p_r)
    n_bl = torch.linalg.cross(p_l, p_b)

    # n_ul = torch.nn.functional.normalize(torch.cross(p_u, p_l), dim=-1)
    # n_ur = torch.nn.functional.normalize(torch.cross(p_r, p_u), dim=-1)
    # n_br = torch.nn.functional.normalize(torch.cross(p_b, p_r), dim=-1)
    # n_bl = torch.nn.functional.normalize(torch.cross(p_l, p_b), dim=-1)

    # n_ul = torch.nn.functional.normalize(torch.cross(p_l, p_u), dim=-1)
    # n_ur = torch.nn.functional.normalize(torch.cross(p_u, p_r), dim=-1)
    # n_br = torch.nn.functional.normalize(torch.cross(p_r, p_b), dim=-1)
    # n_bl = torch.nn.functional.normalize(torch.cross(p_b, p_l), dim=-1)
    
    n = n_ul + n_ur + n_br + n_bl
    n = n[0]
    
    # n *= -torch.sum(camVDir * camN, -1, True).sign() # no cull back

    mask = mask[0, 1:-1, 1:-1, :]

    # n = gaussian_blur(n, filter_size, 1) * mask

    n = torch.nn.functional.normalize(n, dim=-1)
    # n[..., 1] *= -1
    # n *= -1

    n = (n * mask).permute([2, 0, 1])
    return n

def normal2curv(normal, mask):
    # normal = normal.detach()
    n = normal.permute([1, 2, 0])
    m = mask.permute([1, 2, 0])
    n = torch.nn.functional.pad(n[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    m = torch.nn.functional.pad(m[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)
    n_c = (n[:, 1:-1, 1:-1, :]      ) * m[:, 1:-1, 1:-1, :]
    n_u = (n[:,  :-2, 1:-1, :] - n_c) * m[:,  :-2, 1:-1, :]
    n_l = (n[:, 1:-1,  :-2, :] - n_c) * m[:, 1:-1,  :-2, :]
    n_b = (n[:, 2:  , 1:-1, :] - n_c) * m[:, 2:  , 1:-1, :]
    n_r = (n[:, 1:-1, 2:  , :] - n_c) * m[:, 1:-1, 2:  , :]
    curv = (n_u + n_l + n_b + n_r)[0]
    curv = curv.permute([2, 0, 1]) * mask
    curv = curv.norm(1, 0, True)
    return curv

def resize_image(img, factor, mode='bilinear'):
    # print(type(img))
    if factor == 1:
        return img
    is_np = type(img) == np.ndarray
    if is_np:
        resize = torch.from_numpy(img)
    else:
        resize = img.clone()
    dtype = resize.dtype

    if type(factor) == int:
        resize = torch.nn.functional.interpolate(resize[None].to(torch.float32), scale_factor=1/factor, mode=mode)[0].to(dtype)
    elif len(factor) == 2:
        resize = torch.nn.functional.interpolate(resize[None].to(torch.float32), size=factor, mode=mode)[0].to(dtype)
    # else:

    if is_np:
        resize = resize.numpy()
    # print(type(img))
    return resize

def depth2wpos(depth, mask, camera):
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]), indexing='ij')
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)
    
    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2,2])
    Kinv = torch.inverse(K).to(device)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    pose = camera.world_view_transform.to(device)
    Rinv = pose[:3, :3]
    t = pose[3:, :3]
    camWPos = (camPos - t) @ Rinv.t()

    camWPos = (camWPos[..., :3] * mask).permute([2, 0, 1])
    
    return camWPos

def depth2viewDir(depth, camera):
    camD = depth.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]), indexing='ij')
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)
    
    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2,2])
    Kinv = torch.inverse(K).to(device)
    # print(p.shape, Kinv.shape)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)
    center = camera_center(camera)
    viewDir = camPos - center
    viewDir = torch.nn.functional.normalize(viewDir, dim=2).permute([2, 0, 1])
    return viewDir

def camera_center(camera):
    pose = camera.world_view_transform
    Rinv = pose[:3, :3]
    t = pose[3:, :3]
    center = -t @ Rinv.t()
    return center

def world2scrn(xyz, cams, pad):
    utils_mod = load(name="cuda_utils", sources=["utils/ext.cpp", "utils/cuda_utils.cu"])
    device = xyz.device
    mask = [i.get_gtMask().to(device).to(torch.float32) for i in cams]
    mask = [i + utils_mod.contour_padding(i, i.to(torch.bool), pad) for i in mask]
    mask = torch.cat(mask, 0)

    worldPos = xyz#.detach()
    worldPos = torch.cat([worldPos, torch.ones_like(worldPos[:, :1])], 1)[None, :, None]
    
    view_mat = torch.cat([i.world_view_transform[None] for i in cams], 0).to(device)[:, None]
    z_near = torch.cat([torch.tensor([[i.znear]]) for i in cams], 0).to(device)
    z_far = torch.cat([torch.tensor([[i.zfar]]) for i in cams], 0).to(device)

    camPos = (worldPos @ view_mat[..., :3]).squeeze()
    outViewZ = torch.le(camPos[..., 2], z_near) + torch.gt(camPos[..., 2], z_far)

    prj_mat = torch.cat([i.full_proj_transform[None] for i in cams], 0).to(device)[:, None]
    projPos = (worldPos @ prj_mat).squeeze()
    projPos = projPos[..., :3] / (projPos[..., 3:] + 1e-7)

    outViewX = torch.le(projPos[..., 0], -1) + torch.gt(projPos[..., 0], 1)
    outViewY = torch.le(projPos[..., 1], -1) + torch.gt(projPos[..., 1], 1)
    outView = outViewX + outViewY + outViewZ
    # outAllView = torch.all(outView, dim=0)

    reso = torch.cat([torch.tensor([[[i.image_width, i.image_height]]]) for i in cams], 0).to(device)
    prcp = torch.cat([i.prcppoint[None] for i in cams], 0).to(device)[:, None]

    scrnPos = ((projPos[..., :2] + 1) * reso - 1) * 0.5 + reso * (prcp - 0.5)
    ndc = (scrnPos / reso) * 2 - 1

    scrnPos = torch.clip(scrnPos, torch.zeros_like(reso), reso - 1).to(torch.long)

    mask_idx = torch.arange(0, len(mask))[:, None].to(torch.long)

    if mask.mean() == 1:
        inMask = torch.ones_like(outView).to(torch.bool)
    else:
        inMask = mask.permute([0, 2, 1])[mask_idx, scrnPos[..., 0], scrnPos[..., 1]].to(torch.bool)
    # inMaskOrOutView = torch.all(inMask + outView, dim=0)
    # inMaskOrOutView = torch.all(inMask, dim=0)

    # visible = inMaskOrOutView * ~outAllView

    return camPos, ndc, inMask, outView

def resample_points(camera, depth, normal, color, mask):
    camWPos = depth2wpos(depth, mask, camera).permute([1, 2, 0])
    camN = normal.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0]).to(torch.bool)
    mask = mask.detach()[..., 0]
    camN = camN.detach()[mask]
    camWPos = camWPos.detach()[mask]
    camRGB = color.permute([1, 2, 0])[mask]

    Rinv = camera.world_view_transform[:3, :3]

    points = torch.cat([camWPos, camN @ Rinv.t(), camRGB], -1)
    return points

def grid_prune(grid, shift, scale, dim, pts, thrsh=1):
    # print(dim)
    grid_cord = ((pts + shift) * scale).to(torch.long)
    # print(grid_cord.min(), grid_cord.max())
    out = (torch.le(grid_cord, 0) + torch.gt(grid_cord, dim - 1)).any(1)
    # print(grid_cord.min(), grid_cord.max())
    grid_cord = grid_cord.clamp(torch.zeros_like(dim), dim - 1)
    mask = grid[grid_cord[:, 0], grid_cord[:, 1], grid_cord[:, 2]] > thrsh
    mask *= ~out
    # print(grid_cord.shape, mask.shape, mask.sum())
    return mask.to(torch.bool)

def reflect(view, normal):
    return view - 2 * torch.sum(view * normal, 0, True) * normal

def img2video(path, fps=30):
    images = glob.glob(path+'/*.jpg') + glob.glob(path+'/*.png') + glob.glob(path+'/*.JPG')
    images.sort()
    # for i in images[:100]:
    #     print(i)
    # exit()
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{path}/video.mp4', fourcc, fps, (width, height))
    for image in tqdm(images, desc="Creating video"):
        video.write(cv2.imread(image))
    video.release()





