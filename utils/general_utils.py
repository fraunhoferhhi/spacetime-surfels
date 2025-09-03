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
import os
import sys
from datetime import datetime
import numpy as np
import random
from pytorch3d.ops import knn_points
import pymeshlab
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.io import IO
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import open3d as o3d

def cutoff_act(x, low=0.1, high=12):
    return low + (high - low) * torch.sigmoid(x)

def cutoff_act_inverse(x, low=0.1, high=12):
    x_ = (x - low) / (high - low)
    return torch.log(x_ / (1 - x_))

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

import torch.nn.functional as F
def TorchImageResize(image_tensor, resolution):
    """
    Resize a PyTorch tensor image to the given resolution.

    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (C, H, W).
        resolution (tuple): Target resolution as (width, height).

    Returns:
        torch.Tensor: Resized image tensor of shape (C, height, width) 
    """
    assert image_tensor.dtype == torch.uint8
    
    # Current resolution of the image
    current_resolution = (image_tensor.shape[2], image_tensor.shape[1])  # (width, height)

    # Resize only if necessary
    if current_resolution != resolution:
        resized_image = F.interpolate(
            image_tensor.unsqueeze(0),  # Add batch dimension
            size=(resolution[1], resolution[0]),  # Target (height, width)
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        return resized_image
    
    return image_tensor

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    # old_f = sys.stdout
    # class F:
    #     def __init__(self, silent):
    #         self.silent = silent

    #     def write(self, x):
    #         if not self.silent:
    #             if x.endswith("\n"):
    #                 old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
    #             else:
    #                 old_f.write(x)

    #     def flush(self):
    #         old_f.flush()

    # sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def normal2rotation(n):
    # construct a random rotation matrix from normal
    # it would better be positive definite and orthogonal
    n = torch.nn.functional.normalize(n)
    # w0 = torch.rand_like(n)
    w0 = torch.tensor([[1, 0, 0]]).expand(n.shape)
    R0 = w0 - torch.sum(w0 * n, -1, True) * n
    R0 *= torch.sign(R0[:, :1])
    R0 = torch.nn.functional.normalize(R0)
    R1 = torch.linalg.cross(n, R0)
    
    # i = 7859
    # print(R1[i])
    R1 *= torch.sign(R1[:, 1:2]) * torch.sign(n[:, 2:])
    # print(R1[i])
    R = torch.stack([R0, R1, n], -1)
    # print(R[i], torch.det(R).sum(), torch.trace(R[i]))
    q = rotmat2quaternion(R)
    # print(q[i], torch.norm(q[i]))
    # R = quaternion2rotmat(q)
    # print(R[i])
    # for i in range(len(q)):
    #     if torch.isnan(q[i].sum()):
    #         print(i)
    # exit()
    return q

def quaternion2rotmat(q):
    r, x, y, z = q.split(1, -1)
    # R = torch.eye(4).expand([len(q), 4, 4]).to(q.device)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)
    ], -1).reshape([len(q), 3, 3]);
    return R

def rotmat2quaternion(R, normalize=False):
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-6
    r = torch.sqrt(1 + tr) / 2
    # print(torch.sum(torch.isnan(r)))
    q = torch.stack([
        r,
        (R[:, 2, 1] - R[:, 1, 2]) / (4 * r),
        (R[:, 0, 2] - R[:, 2, 0]) / (4 * r),
        (R[:, 1, 0] - R[:, 0, 1]) / (4 * r)
    ], -1)
    if normalize:
        q = torch.nn.functional.normalize(q, dim=-1)
    return q


def knn_pcl(pcl0, pcl1, feat, K):
    nn_dist, nn_idx, nn_vtx = knn_points(pcl0[..., :3][None], pcl1[..., :3][None], K=K+1, return_nn=True)
    nn_dist = nn_dist[0, :, 1:]
    nn_idx = nn_idx[0, :, 1:]
    nn_vtx = nn_vtx[0, :, 1:]
    nn_vtx = torch.mean(nn_vtx, axis=1)
    nn_feat = torch.mean(feat[nn_idx], axis=1)
    return nn_vtx, nn_feat


def compute_obb(vertices):
    # Compute covariance matrix
    cov_matrix = np.cov(vertices, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_idx]
    eigenvalues = eigenvalues[sort_idx]
    
    # Center the vertices
    centroid = np.mean(vertices, axis=0)
    vertices_centered = vertices - centroid
    
    # Rotate vertices to align with eigenvectors
    vertices_transformed = vertices_centered @ eigenvectors
    
    # Compute min and max coordinates in transformed space
    min_coords = np.min(vertices_transformed, axis=0)
    max_coords = np.max(vertices_transformed, axis=0)
    center = centroid
    # right multiply eigenvectors: local bbox to world
    # left multiply eigenvectors: world to local bbox 
    rotation = eigenvectors
    
    return center, rotation, min_coords, max_coords

# def poisson_mesh(path, vtx, normal, color, depth, thrsh):

#     pbar = tqdm(total=4, leave=False)
#     pbar.update(1)
#     pbar.set_description('Poisson meshing')

#     # create pcl with normal from sampled points
#     ms = pymeshlab.MeshSet()
#     pts = pymeshlab.Mesh(vtx.cpu().numpy(), [], normal.cpu().numpy())
#     ms.add_mesh(pts)


#     # poisson reconstruction
#     ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True, samplespernode=1.5)
#     vert = ms.current_mesh().vertex_matrix()
#     face = ms.current_mesh().face_matrix()
#     ms.save_current_mesh(path + '_plain.ply')


#     pbar.update(1)
#     pbar.set_description('Mesh refining')
#     # knn to compute distance and color of poisson-meshed points to sampled points
#     nn_dist, nn_idx, _ = knn_points(torch.from_numpy(vert).to(torch.float32).cuda()[None], vtx.cuda()[None], K=4)
#     nn_dist = nn_dist[0]
#     nn_idx = nn_idx[0].to(color.device)
#     nn_color = torch.mean(color[nn_idx], axis=1)

#     # create mesh with color and quality (distance to the closest sampled points)
#     vert_color = nn_color.clip(0, 1).cpu().numpy()
#     vert_color = np.concatenate([vert_color, np.ones_like(vert_color[:, :1])], 1)
#     ms.add_mesh(pymeshlab.Mesh(vert, face, v_color_matrix=vert_color, v_scalar_array=nn_dist[:, 0].cpu().numpy()))

#     pbar.update(1)
#     pbar.set_description('Mesh cleaning')
#     # prune outlying vertices and faces in poisson mesh
#     ms.compute_selection_by_condition_per_vertex(condselect=f"q>{thrsh}")
#     ms.meshing_remove_selected_vertices()

#     # fill holes
#     ms.meshing_close_holes(maxholesize=300)
#     ms.save_current_mesh(path + '_pruned.ply')

#     # smoothing, correct boundary aliasing due to pruning
#     ms.load_new_mesh(path + '_pruned.ply')
#     ms.apply_coord_laplacian_smoothing(stepsmoothnum=3, boundary=True)
#     ms.save_current_mesh(path + '_pruned.ply')
    
#     pbar.update(1)
#     pbar.close()

def poisson_mesh(path, vtx, normal, color, depth, use_pymeshlab = False):

    pbar = tqdm(total=3)
    pbar.set_description('Poisson meshing & smoothing')

    vtx_np = vtx.cpu().numpy().astype(np.float64)
    normal_np = normal.cpu().numpy().astype(np.float64)
    # color_np = color.cpu().numpy()

    center, rotation, min_coords, max_coords = compute_obb(vtx_np)

    # poisson recon
    if use_pymeshlab:
        # create pcl with normal from sampled points
        ms = pymeshlab.MeshSet()
        pts = pymeshlab.Mesh(vtx_np, [], normal_np)
        ms.add_mesh(pts)
        # poisson reconstruction
        ms.generate_surface_reconstruction_screened_poisson(depth=depth, threads=os.cpu_count() // 2, preclean=True, samplespernode=1.5)
        ms.save_current_mesh(path + '_plain.ply')    
    else: # use open3d
        # Create an open3d PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx_np)
        pcd.normals = o3d.utility.Vector3dVector(normal_np)
        # pcd.colors = o3d.utility.Vector3dVector(color_np)
        # Perform Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth,n_threads=os.cpu_count() // 2)    
        # Save the resulting mesh to a file
        o3d.io.write_triangle_mesh(path + '_plain.ply', mesh)

    # # smoothing, correct boundary aliasing due to pruning
    # ms.apply_coord_laplacian_smoothing(stepsmoothnum=3, boundary=True)

    pbar.update(1)
    # apply color from input points(vtx) to vertices of mesh
    if False:
        pbar.set_description('Mesh coloring')
        vert = ms.current_mesh().vertex_matrix()
        face = ms.current_mesh().face_matrix()
        nn_dist, nn_idx, _ = knn_points(torch.from_numpy(vert).to(torch.float32).cuda()[None], vtx.cuda()[None], K=1) # K=4)
        nn_dist = nn_dist[0]
        nn_idx = nn_idx[0]
        nn_color = torch.mean(color[nn_idx.to(color.device)], axis=1)

        # create mesh with color and quality (distance to the closest sampled points)
        vert_color = nn_color.clip(0, 1).cpu().numpy()
        vert_color = np.concatenate([vert_color, np.ones_like(vert_color[:, :1])], 1)
        ms.add_mesh(pymeshlab.Mesh(vert, face, v_color_matrix=vert_color, v_scalar_array=nn_dist[:, 0].cpu().numpy()))
        ms.save_current_mesh(path + '_colored.ply')
    
    pbar.update(1)    
    pbar.set_description('Mesh cleaning')

    # prune outlying vertices and faces in poisson mesh
    # ms.compute_selection_by_condition_per_vertex(condselect=f"q>{thrsh}")
    # ms.meshing_remove_selected_vertices()

    # fill holes
    # ms.meshing_close_holes(maxholesize=300)

    mesh = o3d.io.read_triangle_mesh(path + '_plain.ply')
    # only for HHI: cut vertices below the floor
    if False:
        bbox_min = np.array([-1e9, 0, -1e9])  # Minimum coordinates of the bounding box
        bbox_max = np.array([1e9, 1e9, 1e9])  # Maximum coordinates of the bounding box
        vertices = np.asarray(mesh.vertices)
        mask = np.invert(np.all((vertices >= bbox_min) & (vertices <= bbox_max), axis=1))
        mesh.remove_vertices_by_mask(mask)    
    # Apply the OBB to the mesh
    mesh_vertices = np.asarray(mesh.vertices)
    vertices_centered = mesh_vertices - center
    vertices_transformed = vertices_centered @ rotation
    buffer = 0 #0.01 # 1cm buffer
    min_coords = min_coords - buffer
    max_coords = max_coords + buffer    
    # Generate mask for vertices inside the OBB
    is_inside = np.all(
        (vertices_transformed >= min_coords) & 
        (vertices_transformed <= max_coords),
        axis=1
    )
    mesh.remove_vertices_by_mask(~is_inside) 

    # if args.smooth_iter > 0:
    #     mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth_iter)

     # only keep largest cluster
    if True:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh.remove_triangles_by_mask(triangles_to_remove)

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    # mesh = mesh.simplify_quadric_decimation(args.n_faces) 
    
    # mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(path + '_cleaned.ply', mesh)
    
    pbar.update(1)
    pbar.close()


def save_pcd(path, vtx, normal, color):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vtx)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud((path + '_pcd.ply'), point_cloud)
    
def custom_sigmoid(x, shift=0.6, scale=30):
    return 1 / (1 + torch.exp(-scale * (x - shift)))
  
def trbfunction(x, flat_top=0): 
    if flat_top == 0:
        return torch.exp(-1*x.pow(2))
    else:
        return custom_sigmoid(torch.exp(-1*x.pow(2)))

def setgtisint8(value):
    print("set current resized gt image as int8 for memory: ", value)
    os.environ['gtisint8'] = str(value)

def getgtisint8():
    #print("get current gt", bool(int(os.getenv('gtisint8'))))
    return bool(int(os.getenv('gtisint8')))