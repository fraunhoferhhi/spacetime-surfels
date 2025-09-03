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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
import torch
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], loader="colmap"):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}


        if loader == "colmap": # colmapvalid only for testing
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.mono_normal, args.use_flow, duration=args.duration, test_views=self.gaussians.test_views, random_init=args.random_init)
        elif loader == "video": # TODO: technicolor and immersive loaders
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.mono_normal, args.use_flow, duration=1, test_views=None)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            json_cams = {
                "train": [],
                "test": []
            }   
            if scene_info.test_cameras:
                for id, cam in enumerate(scene_info.test_cameras):
                    json_cams["test"].append(camera_to_JSON(id, cam)) 
            if scene_info.train_cameras:
                for id, cam in enumerate(scene_info.train_cameras):
                    json_cams["train"].append(camera_to_JSON(id, cam))
            
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file, indent=2)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        print(f"Total views: {len(scene_info.train_cameras)} for training and {len(scene_info.test_cameras)} for testing")
        for resolution_scale in resolution_scales:
            print(f"Loading train views")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
        
            print(f"Loading test views")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if gaussians is None:
            return

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args.dup)
        

        self.gaussians.config.append(False) # camera_lr > 0
        self.gaussians.config = torch.tensor(self.gaussians.config, dtype=torch.float32, device="cuda")

        self.gaussians.start_frame = scene_info.start_frame


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1):
        return self.test_cameras[scale]
    
    def getTrainCamerasByIdx(self, idx, scale=1):
        cameras = self.train_cameras[scale]
        return [cameras[i] for i in idx]
    
    def visualize_cameras(self):
        points = []
        colors = []
        for i in self.getTrainCameras():
            center = i.camera_center.detach().cpu().numpy()
            # print(center)
            viewDir = i.R[:3, 2].cpu().numpy()
            for j in range(1):
                points.append(center + viewDir * j * 0.1)
                # print(center)
                # print(i.T@i.R)
                # colors.append([1, 1, 1, 1.0] if j == 0 else [0, 0, 0, 0.0])
        import pymeshlab
        import numpy as np
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.array(points)))
        ms.save_current_mesh('test/cameras.ply')
        