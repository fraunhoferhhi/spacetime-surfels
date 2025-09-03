# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
from typing import Tuple
import copy
from PIL import Image
import mediapy as media
from tqdm import tqdm
import re
import torch
import trimesh
import pyrender
from utils.graphics_utils import fov2focal
from scene.cameras import Camera
import open3d as o3d
import cv2

def normalize(x: np.ndarray) -> np.ndarray:
  """Normalization helper function."""
  return x / np.linalg.norm(x)

def pad_poses(p: np.ndarray) -> np.ndarray:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = average_pose(poses)
  transform = np.linalg.inv(pad_poses(cam2world))
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform


def average_pose(poses: np.ndarray) -> np.ndarray:
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
  """Construct lookat view matrix."""
  vec2 = normalize(lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
  t = poses[:, :3, 3]
  t_mean = t.mean(axis=0)
  t = t - t_mean

  eigval, eigvec = np.linalg.eig(t.T @ t)
  # Sort eigenvectors in order of largest to smallest eigenvalue.
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  poses_recentered = unpad_poses(transform @ pad_poses(poses))
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

  # Flip coordinate system if z component of y-axis is negative
  if poses_recentered.mean(axis=0)[2, 1] < 0:
    poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    transform = np.diag(np.array([1, -1, -1, 1])) @ transform

  return poses_recentered, transform
  # points = np.random.rand(3,100)
  # points_h = np.concatenate((points,np.ones_like(points[:1])), axis=0)
  # (poses_recentered @ points_h)[0]
  # (transform @ pad_poses(poses) @ points_h)[0,:3]
  # import pdb; pdb.set_trace()

  # # Just make sure it's it in the [-1, 1]^3 cube
  # scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
  # poses_recentered[:, :3, 3] *= scale_factor
  # transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

  # return poses_recentered, transform

def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          const_speed: bool = True,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
  """Generate an elliptical render path based on the given poses."""
  # Calculate the focal point for the path (cameras point toward this).
  center = focus_point_fn(poses)
  # Path height sits at z=0 (in middle of zero-mean capture pattern).
  offset = np.array([center[0], center[1], 0])

  # Calculate scaling for ellipse axes based on input camera positions.
  sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
  # Use ellipse that is symmetric about the focal point in xy.
  low = -sc + offset
  high = sc + offset
  # Optional height variation need not be symmetric
  z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
  z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

  def get_positions(theta):
    # Interpolate between bounds with trig functions to get ellipse in x-y.
    # Optionally also interpolate in z to change camera height along path.
    return np.stack([
        low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
        low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
        z_variation * (z_low[2] + (z_high - z_low)[2] *
                       (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
    ], -1)

  theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
  positions = get_positions(theta)

  #if const_speed:

  # # Resample theta angles so that the velocity is closer to constant.
  # lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
  # theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
  # positions = get_positions(theta)

  # Throw away duplicated last position.
  positions = positions[:-1]

  # Set path's up vector to axis closest to average of input pose up vectors.
  avg_up = poses[:, :3, 1].mean(0)
  avg_up = avg_up / np.linalg.norm(avg_up)
  ind_up = np.argmax(np.abs(avg_up))
  up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

  return np.stack([viewmatrix(p - center, up, p) for p in positions])


def generate_path(viewpoint_cameras, duration, n_frames, intrinsics_cam_idx, extrinsics_cam_idx=None, rotate=False):
  traj = []

  # rotate cams around center
  if rotate: 
    c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
    pose = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
    pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

    # generate new poses
    new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)
    # warp back to orignal scale
    new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)
    intrinsic_cam = viewpoint_cameras[intrinsics_cam_idx]
   
    
    with torch.no_grad():
      for idx, c2w in enumerate(new_poses):
          c2w = c2w @ np.diag([1, -1, -1, 1])
          Rt = np.linalg.inv(c2w)
          R = Rt[:3, :3].T
          T = Rt[:3, 3]
          timestamp = (idx % duration) / duration
          cam = Camera(colmap_id=intrinsic_cam.uid, R=R, T=T, K=intrinsic_cam.K, 
                       FoVx=intrinsic_cam.FoVx, FoVy=intrinsic_cam.FoVy, prcppoint=intrinsic_cam.prcppoint, image=intrinsic_cam.original_image,
                       timestamp=timestamp)
          traj.append(cam)
  
  # fix cam
  else:
    for i in range(n_frames):
      cam = copy.deepcopy(viewpoint_cameras[extrinsics_cam_idx])
      cam.timestamp = (i % duration) / duration
      traj.append(cam)

  return traj

def load_img(pth: str) -> np.ndarray:
  """Load an image and cast to float32."""
  with open(pth, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)
  return image


def create_videos(rendered_dict, video_dir, hw, fps):
  """Creates videos out of the images from dict"""
  video_kwargs = {
      'shape': hw,
      'codec': 'h264',
      'fps': fps,
      'crf': 18,
  }
  
  for k in rendered_dict:
    video_file = os.path.join(video_dir, f'{k}.mp4')
    num_frames = len(rendered_dict[k])
    with media.VideoWriter(video_file, **video_kwargs, input_format='rgb') as writer:
      for idx in tqdm(range(num_frames), desc=f"Writing {k} video"):
        img = rendered_dict[k][idx]
        # img = img / 255.
        if img.dtype != np.uint8:
          img = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
        writer.add_image(img)
        idx += 1


def save_img_u8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')


def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')


def get_largest_mesh_folder(directory):
    # Regular expression to match the desired pattern
    pattern_10 = re.compile(r'poisson_it(\d+)_depth10')
    pattern_9 = re.compile(r'poisson_it(\d+)_depth9')
    
    largest_number = -1
    largest_folder = None

    # Iterate through the files in the directory
    for folder in os.listdir(directory):
        match = pattern_9.match(folder)
        if match:
            number = int(match.group(1))
            if number > largest_number:
                largest_number = number
                largest_folder = folder

    if largest_folder is None: assert False, "cannot find mesh folder"
    return os.path.join(directory, largest_folder)

# ### use pyrender
# def render_mesh(mesh_path, cam, rendered_dict):
#     mesh = trimesh.load(mesh_path)
    
#     # Ensure the mesh has vertex colors to highlight geometry
#     if mesh.visual.vertex_colors is None or len(mesh.visual.vertex_colors) == 0:
#         mesh.visual.vertex_colors = np.ones((mesh.vertices.shape[0], 4), dtype=np.uint8) * 200  # Light grey color

#     # Create a pyrender scene
#     scene = pyrender.Scene()
    
#     # Create a pyrender mesh and add it to the scene
#     render_mesh = pyrender.Mesh.from_trimesh(mesh)
#     scene.add(render_mesh)
    
#     # Define camera intrinsics
#     width, height = cam.image_width, cam.image_height
#     fx, fy = fov2focal(cam.FoVx, width), fov2focal(cam.FoVy, height)
#     cx, cy = cam.prcppoint[0].cpu().numpy() * width, cam.prcppoint[1].cpu().numpy() * height
    
#     # Create a pyrender camera
#     camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
    
#     # Add the camera to the scene with the given pose
#     camera_pose = cam.world_view_transform.T.cpu().numpy()
#     camera_pose[[1, 2]] *= -1
#     camera_pose = np.linalg.inv(camera_pose)
#     scene.add(camera, pose=camera_pose) # pose is c2w in opengl coord. sys.
    
#     # Add a light source to the scene
#     light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
#     scene.add(light, pose=camera_pose)  # Add light at the same position as the camera
    
#     # Create a pyrender offscreen renderer
#     renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    
#     # Render the scene
#     color, _ = renderer.render(scene)
#     rendered_dict["meshes"].append(color)

# using pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    RasterizationSettings,
    DirectionalLights,
    AmbientLights,
    TexturesVertex,
    Materials,  # Import the Materials class to adjust roughness
)
from pytorch3d.utils import cameras_from_opencv_projection
def render_mesh(mesh_path, cam, rendered_dict=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the PLY mesh using Open3D
    o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)

    # Paint the mesh with a uniform color
    vertices = torch.from_numpy(np.asarray(o3d_mesh.vertices)).float().to(device)
    faces = torch.from_numpy(np.asarray(o3d_mesh.triangles)).long().to(device)
    verts_rgb = torch.full([vertices.shape[0], 3], 0.78, device=device)  # Uniform light grey color

    # Convert to a PyTorch3D mesh
    mesh = Meshes(verts=[vertices], faces=[faces], textures=TexturesVertex(verts_features=[verts_rgb]))

    # Set camera parameters
    width, height = cam.image_width, cam.image_height
    # fx, fy = fov2focal(cam.FoVx, width), fov2focal(cam.FoVy, height)
    fx, fy = cam.fx, cam.fy
    cx, cy = cam.prcppoint[0].cpu().numpy() * width, cam.prcppoint[1].cpu().numpy() * height
    cx = float(cx)
    cy = float(cy)

    # Camera intrinsic matrix (K)
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device).unsqueeze(0)

    # Camera rotation (R) and translation (tvec)
    world_to_camera_matrix = cam.world_view_transform.T.cpu().numpy()
    R = torch.tensor(world_to_camera_matrix[:3, :3], device=device).unsqueeze(0)
    tvec = torch.tensor(world_to_camera_matrix[:3, 3], device=device).unsqueeze(0)

    # Image size tensor
    image_size = torch.tensor([[height, width]], device=device)

    # Create PyTorch3D camera using the OpenCV parameters
    cameras = cameras_from_opencv_projection(R=R, tvec=tvec, camera_matrix=K, image_size=image_size)

    # Calculate the direction for the directional light to align with the camera's view direction
    camera_direction = -world_to_camera_matrix[2, :3]  # The third column of the rotation matrix gives the forward direction
    camera_direction = torch.tensor(camera_direction, device=device).unsqueeze(0)

    directional_light = DirectionalLights(
        ambient_color=((0.3, 0.3, 0.3),),  # Match ambient light to ensure base illumination
        diffuse_color=((0.8, 0.8, 0.8),),  # Increase diffuse to brighten the surface with light direction
        specular_color=((0.2, 0.2, 0.2),),  # Increase specular slightly for moderate shine
        direction=camera_direction,  # Align the light direction with the camera
        device=device
    )
    # Adjust material properties to make the surface more reflective but still rough enough
    material = Materials(
        device=device,
        specular_color=((0.2, 0.2, 0.2),),  # A bit higher specular color to simulate moderate shine
        shininess=10.0,  # Increase shininess slightly to make highlights sharper but not too glossy
    )

    # Rasterization settings
    raster_settings = RasterizationSettings(image_size=(height, width), blur_radius=0.0, faces_per_pixel=1)

    # Set up the shader with both ambient and directional lights, and the adjusted material
    shader = SoftPhongShader(device=device, cameras=cameras, lights=directional_light, materials=material)

    # Create the renderer
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=shader)

    # Render the image with both ambient and directional light
    images = renderer(mesh)
    image = images[0, ..., :3].cpu().numpy()

    # Convert the rendered image to a numpy array and store it in the rendered_dict
    image = (image * 255).astype(np.uint8)
    if rendered_dict is not None:
      rendered_dict["meshes"].append(image)
    else: return image

    # Save the image to a file
    # cv2.imwrite("rendered_mesh.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
