import torch
import random
import numpy as np
from scene.gaussian_model import GaussianModel
import open3d as o3d

def get_xyz_at_time(gaussians, time):
    means3D = gaussians.get_xyz
    trbfcenter = gaussians.get_trbfcenter
    pointtimes = torch.ones((gaussians.get_xyz.shape[0],1), dtype=gaussians.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    trbfdistanceoffset = time * pointtimes - trbfcenter
    tforpoly = trbfdistanceoffset.detach()
    for d in range(gaussians.motion_degree):
        means3D = means3D + gaussians._motion[:,3*d:3*(d+1)] * tforpoly ** (d+1)
    return means3D


@torch.no_grad()
def get_high_velocity_frames(dataset, pipe, gaussians, opacity_threshold=0.5):
    """
    Identify frames with high surfel velocities using their 3D flow/velocity at each timestep.
    
    Args:
        dataset: Dataset containing frame information
        pipe: Pipeline object containing configuration parameters
        gaussians: Gaussian surfel object containing position and opacity information
        opacity_threshold: Threshold for considering a surfel as active (default: 0.5)
    
    Returns:
        torch.Tensor: Indices of frames with velocities above the 75th percentile
    """
    frame_velocities = []
    
    for idx in range(dataset.duration):
        t = idx / dataset.duration
        
        # Get opacity values for all surfels at current time
        opacities = gaussians.get_full_opacity(t)
        active_mask = opacities.squeeze() >= opacity_threshold
        
        # Skip frame if no active surfels
        if not torch.any(active_mask):
            frame_velocities.append(0.0)
            continue
            
        # Get velocity vectors for all surfels
        velocities = gaussians.get_flow3D(t)[active_mask]
        velocity_magnitudes = torch.norm(velocities, dim=1)
        
        # Aggregate velocities based on specified method
        if pipe.velocity_aggregation == "average":
            frame_velocity = torch.mean(velocity_magnitudes)
        elif pipe.velocity_aggregation == "median":
            frame_velocity = torch.median(velocity_magnitudes)
        elif pipe.velocity_aggregation == "max":
            frame_velocity = torch.max(velocity_magnitudes)
        else:
            raise ValueError(f"Unknown velocity aggregation method: {pipe.velocity_aggregation}")
            
        frame_velocities.append(frame_velocity.item())
    
    # Identify frames with high velocities (above 75th percentile)
    velocities_tensor = torch.tensor(frame_velocities)
    threshold = torch.quantile(velocities_tensor, 0.90)
    high_velocity_frames = torch.nonzero(velocities_tensor >= threshold).squeeze()
    
    return high_velocity_frames
        
# Taken from E-D3DGS https://github.com/JeongminB/E-D3DGS/blob/8cd58f3b7bef62c31d976d669cde8be34af46253/utils/extra_utils.py#L6C1-L20C65
def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)

def knn_self_pytorch_batched(pcd, k, batch_size=1024):
    """
    Batched k-Nearest Neighbors using PyTorch tensors, excluding the query point itself.
    
    Args:
        pcd (torch.Tensor): Tensor of shape (n_pt, n_features), the points for which neighbors are to be found.
        k (int): Number of neighbors to find (excluding the query point itself).
        batch_size (int): Batch size for processing pairwise distances.
        
    Returns:
        torch.Tensor: Indices of the k nearest neighbors for each point, shape (n_pt, k).
        torch.Tensor: Distances to the k nearest neighbors for each point, shape (n_pt, k).
    """
    device = pcd.device

    n_points = pcd.size(0)
    distances_all = torch.empty((n_points, k), dtype=pcd.dtype, device=device)
    indices_all = torch.empty((n_points, k), dtype=torch.long, device=device)

    # Precompute norms
    norms = torch.sum(pcd**2, dim=1)  # (n_pt,)

    for i in range(0, n_points, batch_size):
        # Process batch
        start = i
        end = min(i + batch_size, n_points)

        # Compute pairwise distances for the batch
        batch_pcd = pcd[start:end]  # (batch_size, n_features)
        batch_norms = torch.sum(batch_pcd**2, dim=1).unsqueeze(1)  # (batch_size, 1)
        distances = (
            batch_norms
            + norms.unsqueeze(0)  # (1, n_points)
            - 2 * batch_pcd @ pcd.T  # (batch_size, n_points)
        )
        distances = torch.clamp(distances, min=0)

        # Exclude the query point itself by setting diagonal to infinity
        indices = torch.arange(start, end, device=device)
        distances[torch.arange(distances.size(0)), indices] = float('inf')

        # Find k nearest neighbors for the batch
        batch_distances, batch_indices = torch.topk(
            distances, k, dim=1, largest=False, sorted=True
        )

        # Store results
        distances_all[start:end] = batch_distances
        indices_all[start:end] = batch_indices

    return distances_all, indices_all
