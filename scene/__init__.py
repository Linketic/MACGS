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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_macgs import GaussianModelMACGS
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from scene.dataset_readers import add_points, SceneInfo

try:
    from diff_gaussian_rasterization_macgs._C import sphere_ellipsoid_intersection, allocate_minimum_redundancy_value, find_minimum_projected_pixel_size
    from simple_knn._C import distIndex2
except:
    from diff_gaussian_rasterization_reduced_3dgs._C import sphere_ellipsoid_intersection, allocate_minimum_redundancy_value, find_minimum_projected_pixel_size
    from simple_knn_reduced_3dgs._C import distIndex2


class Scene:

    # gaussians : GaussianModel
    gaussians: GaussianModelMACGS

    def __init__(self, args : ModelParams, gaussians : GaussianModelMACGS, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False):
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
        self.video_cameras = {}
        scene_info: SceneInfo
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
            dataset_type="colmap"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.extension)
            dataset_type="blender"
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            dataset_type="dynerf"
        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
            dataset_type="nerfies"
        elif os.path.exists(os.path.join(args.source_path,"train_meta.json")):
            scene_info = sceneLoadTypeCallbacks["PanopticSports"](args.source_path)
            dataset_type="PanopticSports"
        elif os.path.exists(os.path.join(args.source_path,"points3D_multipleview.ply")):
            scene_info = sceneLoadTypeCallbacks["MultipleView"](args.source_path)
            dataset_type="MultipleView"
        else:
            assert False, "Could not recognize scene type!"
        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        # self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        if args.add_points:
            print("add points.")
            # breakpoint()
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))
        self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)
        self.gaussians._student_deformation.deformation_net.set_aabb(xyz_max, xyz_min)
        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply"
                )
            )
            self.gaussians.load_model(
                os.path.join(
                    self.model_path,
                    "point_cloud", 
                    "iteration_" + str(self.loaded_iter)
                )
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
    
    def getTrainCameras(self, scale=1.0, load_image=True):
        self.train_camera.load_image = load_image
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera
    

class SceneMACGS(Scene):
    
    def __init__(self, args: ModelParams, gaussians: GaussianModelMACGS, load_iteration=None, shuffle=True, resolution_scales=[1], load_coarse=False, model="quantised_half"):
        models_configuration = {
            'baseline': {
                'quantised': False,
                'half_float': False,
                'name': 'point_cloud.ply'
                },
            'quantised': {
                'quantised': True,
                'half_float': False,
                'name': 'point_cloud_quantised.ply'
                },
            'quantised_half': {
                'quantised': True,
                'half_float': True,
                'name': 'point_cloud_quantised_half.ply'
                },
        }
        super().__init__(args, gaussians, load_iteration, shuffle, resolution_scales, load_coarse)
        if load_iteration:
            name = models_configuration[model]["name"]
            print(f"Loading trained model at iteration {self.loaded_iter}, use: {name}")
            self.gaussians.load_ply(
                os.path.join(args.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), name),
                quantised=models_configuration[model]["quantised"],
                half_float=models_configuration[model]["half_float"]
            )
        
        self._camera_cache = {
            'camera_centers': [],
            'view_transforms': [],
            'proj_transforms': [],
            'inv_proj_transforms': [],
            'FoVx': [],
            'FoVy': [],
            'heights': [],
            'widths': [],
        }
        self._cache_initialized = False

    @property
    def camera_params(self):
        if not self._cache_initialized:
            self._initialize_camera_cache()
        return self._camera_cache

    def save(self, iteration, stage, quantise=False, half_float=False):
        ply_name = "point_cloud"
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            if quantise:
                ply_name += "_quantised"
            if half_float:
                ply_name += "_half"
        ply_name += ".ply"
        self.gaussians.save_ply(os.path.join(point_cloud_path, ply_name), quantise, half_float)
        self.gaussians.save_deformation(point_cloud_path)

    def _initialize_camera_cache(self):
        if not self._cache_initialized:
            print("Getting camera parameters, this may take a while...")
            # cameras = self.getTrainCameras(load_image=False)
            cameras = self.getTrainCameras()
            num_cameras = len(cameras)

            for key in self._camera_cache.keys():
                self._camera_cache[key] = [None] * num_cameras

            from concurrent.futures import ThreadPoolExecutor

            def process_camera(idx, camera):
                self._camera_cache['camera_centers'][idx] = camera.camera_center
                self._camera_cache['view_transforms'][idx] = camera.world_view_transform
                self._camera_cache['proj_transforms'][idx] = camera.full_proj_transform
                self._camera_cache['inv_proj_transforms'][idx] = camera.inverse_full_proj_transform
                self._camera_cache['FoVx'][idx] = camera.FoVx
                self._camera_cache['FoVy'][idx] = camera.FoVy
                self._camera_cache['heights'][idx] = camera.image_height
                self._camera_cache['widths'][idx] = camera.image_width

            with ThreadPoolExecutor() as executor:
                list(executor.map(lambda x: process_camera(*x), enumerate(cameras)))

            self._cache_initialized = True

    def invalidate_camera_cache(self):
        self._cache_initialized = False
        self._camera_cache = {
            'camera_centers': [],
            'view_transforms': [],
            'proj_transforms': [],
            'inv_proj_transforms': [],
            'FoVx': [],
            'FoVy': [],
            'heights': [],
            'widths': [],
        }

    def calculate_redundancy_metric(self, pixel_scale=1.0, num_neighbours=30):
        if not self._cache_initialized:
            self._initialize_camera_cache()
        
        # Get minimum projected pixel size
        cube_size = find_minimum_projected_pixel_size(
            torch.stack(self.camera_params['proj_transforms'], dim=0).to("cuda"),
            torch.stack(self.camera_params['inv_proj_transforms'], dim=0).to("cuda"),
            self.gaussians._xyz,
            torch.tensor(self.camera_params['heights'], device="cuda", dtype=torch.int32),
            torch.tensor(self.camera_params['widths'], device="cuda", dtype=torch.int32)
        )
        
        scaled_pixel_size = cube_size * pixel_scale
        half_diagonal = scaled_pixel_size * torch.sqrt(torch.tensor([3], device="cuda")) / 2 
        
        # Find neighbours as candidates for the intersection test 找到高斯点的30个最近邻
        _, indices = distIndex2(self.gaussians.get_xyz, num_neighbours)
        indices = indices.view(-1, num_neighbours)  # 调整为 [num_primitives, num_neighbours]
        
        # Do the intersection check
        redundancy_metrics, intersection_mask = sphere_ellipsoid_intersection(self.gaussians._xyz,
                                                                              self.gaussians.get_scaling,
                                                                              self.gaussians.get_rotation,
                                                                              indices,
                                                                              half_diagonal,
                                                                              num_neighbours)
        # We haven't counted count for the primitive at the center of each sphere, so add 1 to everything
        redundancy_metrics += 1
        
        indices = torch.cat((torch.arange(self.gaussians.num_primitives, device="cuda", dtype=torch.int).view(-1, 1), indices), dim=1)      # 将高斯点自身索引加入到最近邻索引的第一列
        intersection_mask = torch.cat((torch.ones_like(self.gaussians._opacity, device="cuda", dtype=bool), intersection_mask), dim=1)    # 将每个高斯点的自身交叉掩码(True)添加到邻居交叉掩码第一列
        
        min_redundancy_metrics = allocate_minimum_redundancy_value(redundancy_metrics, indices, intersection_mask, num_neighbours+1)[0]
        return min_redundancy_metrics, cube_size