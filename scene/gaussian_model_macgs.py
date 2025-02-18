import os
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement

from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, pcast_i16_to_f32
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera

from typing import Optional, List
from collections import OrderedDict

try:
    from diff_gaussian_rasterization_macgs._C import calculate_colours_variance, kmeans_cuda
except:
    from diff_gaussian_rasterization_reduced_3dgs._C import calculate_colours_variance, kmeans_cuda

class Codebook():
    def __init__(self, ids, centers):
        self.ids = ids
        self.centers = centers
    
    def evaluate(self):
        return self.centers[self.ids.flatten().long()]

def generate_codebook(values, inverse_activation_fn=lambda x: x, num_clusters=256, tol=0.0001):
    shape = values.shape
    values = values.flatten().view(-1, 1)
    centers = values[torch.randint(values.shape[0], (num_clusters, 1), device="cuda").squeeze()].view(-1,1)

    ids, centers = kmeans_cuda(values, centers.squeeze(), tol, 500)
    ids = ids.byte().squeeze().view(shape)
    centers = centers.view(-1,1)

    return Codebook(ids.cuda(), inverse_activation_fn(centers.cuda()))

class GaussianModelMACGS(GaussianModel):
    
    def __init__(self, sh_degree: int, args, variable_sh_bands: bool=False):
        super().__init__(sh_degree, args)
        self.variable_sh_bands = variable_sh_bands
        self._degrees = torch.empty(0)
        self._codebook_dict = None
        self._splatted_num_accum = torch.empty(0)   # 冗余度指标

    def capture(self):
        return (
            *(super().capture()),
            self._degrees,
        )
    
    def restore(self, model_args, training_args):
        (
            self.active_sh_degree, 
            self._xyz, 
            deform_state,
            student_deform_state,
            self._deformation_table,
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            self._degrees
        ) = model_args
        self._deformation.load_state_dict(deform_state)
        self._student_deformation.load_state_dict(student_deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    
    @property
    def per_band_count(self):
        result = list()
        if self.variable_sh_bands:
            for tensor in self._features_rest:
                result.append(tensor.shape[0])
        return result
    
    @property
    def num_primitives(self):
        return self._xyz.shape[0]
    
    @property
    def get_features(self):
        if self.variable_sh_bands:
            features = list()
            index_start = 0
            for idx, sh_tensor in enumerate(self._features_rest):
                index_end = index_start + self.per_band_count[idx]
                features.append(torch.cat((self._features_dc[index_start: index_end], sh_tensor), dim=1))
                index_start = index_end
        else:
            features = torch.cat((self._features_dc, self._features_rest), dim=1)
        return features
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self._degrees += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, time_line: int):
        super().create_from_pcd(pcd, spatial_lr_scale, time_line)
        self._degrees = torch.zeros((self.num_primitives, 1), device='cuda', dtype=torch.int32)
    
    def construct_list_of_attributes(self, rest_coeffs_num=45):
        # l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        for i in range(rest_coeffs_num):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_ply(self, path, quantised=False, half_float=False):
        float_type = 'int16' if half_float else 'f4'
        attribute_type = 'u1' if quantised else float_type
        max_sh_coeffs = (self.max_sh_degree + 1) ** 2 - 1
        mkdir_p(os.path.dirname(path))
        elements_list = []

        if quantised:
            # Read codebook dict to extract ids and centers
            if self._codebook_dict is None:
                print("Clustering codebook missing. Returning without saving")
                return

            opacity = self._codebook_dict["opacity"].ids
            scaling = self._codebook_dict["scaling"].ids
            rot = torch.cat((self._codebook_dict["rotation_re"].ids,
                            self._codebook_dict["rotation_im"].ids),
                            dim=1)
            features_dc = self._codebook_dict["features_dc"].ids
            features_rest = torch.stack([self._codebook_dict[f"features_rest_{i}"].ids
                                        for i in range(max_sh_coeffs)
                                        ], dim=1).squeeze()

            dtype_full = [(k, float_type) for k in self._codebook_dict.keys()]
            codebooks = np.empty(256, dtype=dtype_full)

            centers_numpy_list = [v.centers.detach().cpu().numpy() for v in self._codebook_dict.values()]

            if half_float:
                # No float 16 for plydata, so we just pointer cast everything to int16
                for i in range(len(centers_numpy_list)):
                    centers_numpy_list[i] = np.cast[np.float16](centers_numpy_list[i]).view(dtype=np.int16)
                
            codebooks[:] = list(map(tuple, np.concatenate([ar for ar in centers_numpy_list], axis=1)))
                
        else:
            opacity = self._opacity
            scaling = self._scaling
            rot = self._rotation
            features_dc = self._features_dc
            features_rest = self._features_rest

        for sh_degree in range(self.max_sh_degree + 1):
            coeffs_num = (sh_degree+1)**2 - 1
            degrees_mask = (self._degrees == sh_degree).squeeze()

            #  Position is not quantised
            if half_float:
                xyz = self._xyz[degrees_mask].detach().cpu().half().view(dtype=torch.int16).numpy()
            else:
                xyz = self._xyz[degrees_mask].detach().cpu().numpy()

            f_dc = features_dc[degrees_mask].detach().contiguous().cpu().view(-1,3).numpy()
            # Transpose so that to save rest featrues as rrr ggg bbb instead of rgb rgb rgb
            if self.variable_sh_bands:
                f_rest = features_rest[sh_degree][:, :coeffs_num].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            else:
                f_rest = features_rest[degrees_mask][:, :coeffs_num].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = opacity[degrees_mask].detach().cpu().numpy()
            scale = scaling[degrees_mask].detach().cpu().numpy()
            rotation = rot[degrees_mask].detach().cpu().numpy()

            dtype_full = [(attribute, float_type) 
                          if attribute in ['x', 'y', 'z'] else (attribute, attribute_type) 
                          for attribute in self.construct_list_of_attributes(coeffs_num * 3)]
            elements = np.empty(degrees_mask.sum(), dtype=dtype_full)

            attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scale, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            elements_list.append(PlyElement.describe(elements, f'vertex_{sh_degree}'))
        if quantised:
            elements_list.append(PlyElement.describe(codebooks, f'codebook_centers'))
        PlyData(elements_list).write(path)

    def _parse_vertex_group(self,
                            vertex_group,
                            sh_degree,
                            float_type,
                            attribute_type,
                            max_coeffs_num,
                            quantised,
                            half_precision,                            
                            codebook_centers_torch=None):
        coeffs_num = (sh_degree+1)**2 - 1
        num_primitives = vertex_group.count

        xyz = np.stack((np.asarray(vertex_group["x"], dtype=float_type),
                        np.asarray(vertex_group["y"], dtype=float_type),
                        np.asarray(vertex_group["z"], dtype=float_type)), axis=1)

        opacity = np.asarray(vertex_group["opacity"], dtype=attribute_type)[..., np.newaxis]
    
        # Stacks the separate components of a vector attribute into a joint numpy array
        # Defined just to avoid visual clutter
        def stack_vector_attribute(name, count):
            return np.stack([np.asarray(vertex_group[f"{name}_{i}"], dtype=attribute_type)
                            for i in range(count)], axis=1)

        features_dc = stack_vector_attribute("f_dc", 3).reshape(-1, 1, 3)
        scaling = stack_vector_attribute("scale", 3)
        rotation = stack_vector_attribute("rot", 4)
        
        # Take care of error when trying to stack 0 arrays
        if sh_degree > 0:
            features_rest = stack_vector_attribute("f_rest", coeffs_num*3).reshape((num_primitives, 3, coeffs_num))
        else:
            features_rest = np.empty((num_primitives, 3, 0), dtype=attribute_type)

        if not self.variable_sh_bands:
            # Using full tensors (P x 15) even for points that don't require it
            features_rest = np.concatenate(
                (features_rest,
                    np.zeros((num_primitives, 3, max_coeffs_num - coeffs_num), dtype=attribute_type)), axis=2)

        degrees = np.ones(num_primitives, dtype=np.int32)[..., np.newaxis] * sh_degree

        xyz = torch.from_numpy(xyz).cuda()
        if half_precision:
            xyz = pcast_i16_to_f32(xyz)
        features_dc = torch.from_numpy(features_dc).contiguous().cuda()
        features_rest = torch.from_numpy(features_rest).contiguous().cuda()
        opacity = torch.from_numpy(opacity).cuda()
        scaling = torch.from_numpy(scaling).cuda()
        rotation = torch.from_numpy(rotation).cuda()
        degrees = torch.from_numpy(degrees).cuda()

        # If quantisation has been used, it is needed to index the centers
        if quantised:
            features_dc = codebook_centers_torch['features_dc'][features_dc.view(-1).long()].view(-1, 1, 3)

            # This is needed as we might have padded the features_rest tensor with zeros before
            reshape_channels = coeffs_num if self.variable_sh_bands else max_coeffs_num            
            # The gather operation indexes a 256x15 tensor with a (P*3)features_rest index tensor,
            # in a column-wise fashion
            # Basically this is equivalent to indexing a single codebook with a P*3 index
            # features_rest times inside a loop
            features_rest = codebook_centers_torch['features_rest'].gather(0, features_rest.view(num_primitives*3, reshape_channels).long()).view(num_primitives, 3, reshape_channels)
            opacity = codebook_centers_torch['opacity'][opacity.long()]
            scaling = codebook_centers_torch['scaling'][scaling.view(num_primitives*3).long()].view(num_primitives, 3)
            # Index the real and imaginary part separately
            rotation = torch.cat((
                codebook_centers_torch['rotation_re'][rotation[:, 0:1].long()],
                codebook_centers_torch['rotation_im'][rotation[:, 1:].reshape(num_primitives*3).long()].view(num_primitives,3)
                ), dim=1)

        return {
            'xyz': xyz,
            'opacity': opacity,
            'features_dc': features_dc,
            'features_rest': features_rest,
            'scaling': scaling,
            'rotation': rotation,
            'degrees': degrees
        }

    def load_ply(self, path, half_float=False, quantised=False):
        plydata = PlyData.read(path)

        xyz_list = []
        features_dc_list = []
        features_rest_list = []
        opacity_list = []
        scaling_list = []
        rotation_list = []
        degrees_list = []

        float_type = 'int16' if half_float else 'f4'
        attribute_type = 'u1' if quantised else float_type
        max_coeffs_num = (self.max_sh_degree+1)**2 - 1

        codebook_centers_torch = None
        if quantised:
            # Parse the codebooks.
            # The layout is 256 x 20, where 256 is the number of centers and 20 number of codebooks
            # In the future we could have different number of centers
            codebook_centers = plydata.elements[-1]

            codebook_centers_torch = OrderedDict()
            codebook_centers_torch['features_dc'] = torch.from_numpy(np.asarray(codebook_centers['features_dc'], dtype=float_type)).cuda()
            codebook_centers_torch['features_rest'] = torch.from_numpy(np.concatenate(
                [
                    np.asarray(codebook_centers[f'features_rest_{i}'], dtype=float_type)[..., np.newaxis]
                    for i in range(max_coeffs_num)
                ], axis=1)).cuda()
            codebook_centers_torch['opacity'] = torch.from_numpy(np.asarray(codebook_centers['opacity'], dtype=float_type)).cuda()
            codebook_centers_torch['scaling'] = torch.from_numpy(np.asarray(codebook_centers['scaling'], dtype=float_type)).cuda()
            codebook_centers_torch['rotation_re'] = torch.from_numpy(np.asarray(codebook_centers['rotation_re'], dtype=float_type)).cuda()
            codebook_centers_torch['rotation_im'] = torch.from_numpy(np.asarray(codebook_centers['rotation_im'], dtype=float_type)).cuda()

            # If use half precision then we have to pointer cast the int16 to float16
            # and then cast them to floats, as that's the format that our renderer accepts
            if half_float:
                for k, v in codebook_centers_torch.items():
                    codebook_centers_torch[k] = pcast_i16_to_f32(v)

        # Iterate over the point clouds that are stored on top level of plyfile
        # to get the various fields values 
        for sh_degree in range(0, self.max_sh_degree+1):
            attribues_dict = self._parse_vertex_group(plydata.elements[sh_degree],
                                                      sh_degree,
                                                      float_type,
                                                      attribute_type,
                                                      max_coeffs_num,
                                                      quantised,
                                                      half_float,
                                                      codebook_centers_torch)

            xyz_list.append(attribues_dict['xyz'])
            features_dc_list.append(attribues_dict['features_dc'])
            features_rest_list.append(attribues_dict['features_rest'].transpose(1,2))
            opacity_list.append(attribues_dict['opacity'])
            scaling_list.append(attribues_dict['scaling'])
            rotation_list.append(attribues_dict['rotation'])
            degrees_list.append(attribues_dict['degrees'])

        # Concatenate the tensors into one, to be used in our program
        # TODO: allow multiple PCDs to be rendered/optimise and skip this step
        xyz = torch.cat((xyz_list), dim=0)
        features_dc = torch.cat((features_dc_list), dim=0)
        if not self.variable_sh_bands:
            features_rest = torch.cat((features_rest_list), dim=0)
        else:
            features_rest = features_rest_list
        opacity = torch.cat((opacity_list), dim=0)
        scaling = torch.cat((scaling_list), dim=0)
        rotation = torch.cat((rotation_list), dim=0)
        
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        if not self.variable_sh_bands:
            self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        else:
            for tensor in features_rest_list:
                tensor.requires_grad_(True)
            self._features_rest = features_rest_list
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._degrees = torch.cat((degrees_list), dim=0)

        self.active_sh_degree = self.max_sh_degree

    def produce_clusters(self, num_clusters=256, store_dict_path=None):
        max_coeffs_num = (self.max_sh_degree + 1)**2 - 1
        codebook_dict = OrderedDict({})

        codebook_dict["features_dc"] = generate_codebook(self._features_dc.detach()[:, 0],
                                                         num_clusters=num_clusters, tol=0.001)
        for sh_degree in range(max_coeffs_num):
                codebook_dict[f"features_rest_{sh_degree}"] = generate_codebook(
                    self._features_rest.detach()[:, sh_degree], num_clusters=num_clusters)

        codebook_dict["opacity"] = generate_codebook(self.get_opacity.detach(),
                                                     self.inverse_opacity_activation, num_clusters=num_clusters)
        codebook_dict["scaling"] = generate_codebook(self.get_scaling.detach(),
                                                     self.scaling_inverse_activation, num_clusters=num_clusters)
        codebook_dict["rotation_re"] = generate_codebook(self.get_rotation.detach()[:, 0:1],
                                                         num_clusters=num_clusters)
        codebook_dict["rotation_im"] = generate_codebook(self.get_rotation.detach()[:, 1:],
                                                         num_clusters=num_clusters)
        if store_dict_path is not None:
            torch.save(codebook_dict, os.path.join(store_dict_path, 'codebook.pt'))
        
        self._codebook_dict = codebook_dict

    def apply_clustering(self, codebook_dict=None):
        max_coeffs_num = (self.max_sh_degree + 1)**2 - 1
        if codebook_dict is None:
            return

        opacity = codebook_dict["opacity"].evaluate().requires_grad_(True)
        scaling = codebook_dict["scaling"].evaluate().view(-1, 3).requires_grad_(True)
        rotation = torch.cat((codebook_dict["rotation_re"].evaluate(),
                            codebook_dict["rotation_im"].evaluate().view(-1, 3)),
                            dim=1).squeeze().requires_grad_(True)
        features_dc = codebook_dict["features_dc"].evaluate().view(-1, 1, 3).requires_grad_(True)
        features_rest = []
        for sh_degree in range(max_coeffs_num):
            features_rest.append(codebook_dict[f"features_rest_{sh_degree}"].evaluate().view(-1, 3))

        features_rest = torch.stack([*features_rest], dim=1).squeeze().requires_grad_(True)

        with torch.no_grad():
            self._opacity = opacity
            self._scaling = scaling
            self._rotation = rotation
            self._features_dc = features_dc
            self._features_rest = features_rest

    def prune_points(self, mask, store_grads=False):
        super().prune_points(mask, store_grads)
        self._degrees = self._degrees[~mask]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table, new_degrees, store_grads=False):
        super().densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table, store_grads)
        self._degrees = torch.cat((self._degrees, new_degrees), dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, store_grads=False):
        # The pads are used to ignore the densification that happened before (densify_and_clone)
        n_init_points = self.num_primitives
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        n_points_split = selected_pts_mask.sum().item()

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)  # 选择满足条件的点的缩放值, 重复 N=2 次
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)  # 选择满足条件的点的旋转值, 重复 N=2 次
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1) # 通过旋转和位移生成新的点
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))    # 计算新的缩放值
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        new_degrees = self._degrees[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_deformation_table, new_degrees)   # 将新的点添加到模型中

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, store_grads)
        return n_points_split
    
    def densify_and_clone(self, grads, grad_threshold, scene_extent, density_threshold=20, displacement_scale=20, model_path=None, iteration=None, stage=None, store_grads=False):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = grads.squeeze() >= grad_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)     # 使得选择的点的缩放值小于一定的阈值
        
        n_points_cloned = selected_pts_mask.sum().item()
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]
        new_degrees = self._degrees[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table, new_degrees, store_grads)
        return n_points_cloned
    
    def _low_variance_colour_culling(self, threshold, weighted_variance, weighted_mean):
        original_degrees = torch.zeros_like(self._degrees)
        original_degrees.copy_(self._degrees)

        # Uniform colour culling
        weighted_colour_std = weighted_variance.sqrt()
        weighted_colour_std[weighted_colour_std.isnan()] = 0
        weighted_colour_std = weighted_colour_std.mean(dim=2).squeeze()

        std_mask = weighted_colour_std < threshold
        self._features_dc[std_mask] = (weighted_mean[std_mask] - 0.5) / 0.28209479177387814 # 0.28209479177387814 = 1 / (sqrt(pi) * 2) 球谐函数 0 阶系数
        self._degrees[std_mask] = 0
        self._features_rest[std_mask] = 0

    def _low_distance_colour_culling(self, threshold, colour_distances):
        colour_distances[colour_distances.isnan()] = 0

        # Loop from active_sh_degree - 1 to 0, since the comparisons
        # are always done based on the max band that corresponds to active_sh_degree
        for sh_degree in range(self.active_sh_degree - 1, 0, -1):
            coeffs_num = (sh_degree+1)**2 - 1   # 该阶次的球谐函数系数个数
            mask = colour_distances[:, sh_degree] < threshold
            self._degrees[mask] = torch.min(
                    torch.tensor([sh_degree], device="cuda", dtype=int),
                    self._degrees[mask]
                ).int()
            
            # Zero-out the associated SH coefficients for clarity,
            # as they won't be used in rasterisation due to the degrees field
            self._features_rest[mask, coeffs_num:] = 0

    def cull_sh_bands(self, camera_parameters, threshold=0*np.sqrt(3)/255, std_threshold=0.):
        device = "cuda"
        camera_centers = torch.stack(camera_parameters['camera_centers'], dim=0).to(device)
        view_transforms = torch.stack(camera_parameters['view_transforms'], dim=0).to(device)
        proj_transforms = torch.stack(camera_parameters['proj_transforms'], dim=0).to(device)
        FoVx = torch.tensor(camera_parameters['FoVx'], device=device, dtype=torch.float32)
        FoVy = torch.tensor(camera_parameters['FoVy'], device=device, dtype=torch.float32)
        heights = torch.tensor(camera_parameters['heights'], device=device, dtype=torch.int32)
        widths = torch.tensor(camera_parameters['widths'], device=device, dtype=torch.int32)

        # Wrapping in a function since it's called with the same parameters twice 计算颜色方差和加权平均
        def run_calculate_colours_variance():
            return calculate_colours_variance(
                camera_centers,
                self.get_xyz,
                self._opacity,
                self.get_scaling,
                self.get_rotation,
                view_transforms,
                proj_transforms,
                torch.tan(FoVx*0.5),
                torch.tan(FoVy*0.5),
                heights,
                widths,
                self.get_features,
                self._degrees,
                self.active_sh_degree)
        
        _, weighted_variance, weighted_mean = run_calculate_colours_variance()
        self._low_variance_colour_culling(std_threshold, weighted_variance, weighted_mean)  # 根据标准差阈值 std_threshold 修剪颜色方差较低的点。

        # Recalculate to account for the changed values
        colour_distances, _, _ = run_calculate_colours_variance()
        self._low_distance_colour_culling(threshold, colour_distances)  # 根据颜色距离阈值 threshold 修剪颜色距离较低的点。

    def mercy_points(self, densification_statistics_dict, lambda_mercy=2, mercy_minimum=2, mercy_type='redundancy_opacity'):
        mean = self._splatted_num_accum.squeeze().float().mean(dim=0, keepdim=True)
        std = self._splatted_num_accum.squeeze().float().std(dim=0, keepdim=True)

        threshold = max((mean + lambda_mercy * std).item(), mercy_minimum)

        mask = (self._splatted_num_accum > threshold).squeeze()

        if mercy_type == 'redundancy_opacity':
            # Prune redundant points based on 50% lowest opacity
            mask[mask.clone()] = self.get_opacity[mask].squeeze() < self.get_opacity[mask].median()
        elif mercy_type == 'redundancy_random':
            # Prune 50% redundant points at random
            mask[mask.clone()] = torch.rand(mask[mask].shape, device="cuda").squeeze() < 0.5
        elif mercy_type == 'opacity':
            # Prune based just on opacity
            threshold = self.get_opacity.quantile(0.045)
            mask = (self.get_opacity < threshold).squeeze()
        elif mercy_type == 'redundancy_opacity_opacity':
            # Prune based on opacity and on redundancy + opacity (options 1 and 3)
            mask[mask.clone()] = self.get_opacity[mask].squeeze() < self.get_opacity[mask].median() # mask[mask.clone()] 的对已经筛选出的冗余度超过阈值的点进行进一步筛选, 标记不透明底低于中位数的点
            threshold = torch.min(self.get_opacity.quantile(0.03), torch.tensor([0.05], device="cuda")) # 取不透明度的0.03分位数和0.05的最小值
            mask = torch.logical_or(mask, (self.get_opacity < threshold).squeeze())
        elif mercy_type == "dynamic_static_mercy":
            # Prune based on dynamic/static points
            dynamic_mask = self._deformation_table
            static_mask = ~dynamic_mask

            # 对动态点: 使用较松的阈值,只有冗余度非常高时才筛选
            dynamic_threshold = max((mean + lambda_mercy * 2 * std).item(), mercy_minimum * 2)
            dynamic_redundant = (self._splatted_num_accum > dynamic_threshold).squeeze()
            dynamic_redundant = torch.logical_and(dynamic_redundant, dynamic_mask)

            # 对动态点根据不透明度筛选,但使用较松的阈值
            dynamic_redundant[dynamic_redundant.clone()] = self.get_opacity[dynamic_redundant].squeeze() < self.get_opacity[dynamic_redundant].median() * 0.8

            # 对静态点:使用正常的冗余度阈值
            static_redundant = (self._splatted_num_accum > threshold).squeeze()
            static_redundant = torch.logical_and(static_redundant, static_mask)

            # 对静态点根据不透明度筛选
            static_redundant[static_redundant.clone()] = self.get_opacity[static_redundant].squeeze() < self.get_opacity[static_redundant].median()

            mask = torch.logical_or(dynamic_redundant, static_redundant)

            # 额外的不透明度筛选,对动静点使用不同阈值
            dynamic_opacity_threshold = torch.min(self.get_opacity.quantile(0.01), torch.tensor([0.02], device="cuda"))
            static_opacity_threshold = torch.min(self.get_opacity.quantile(0.03), torch.tensor([0.05], device="cuda"))

            dynamic_opactity_mask = torch.logical_and((self.get_opacity < dynamic_opacity_threshold).squeeze(), dynamic_mask)
            static_opacity_mask = torch.logical_and((self.get_opacity < static_opacity_threshold).squeeze(), static_mask)
            mask = torch.logical_or(mask, torch.logical_or(dynamic_opactity_mask, static_opacity_mask))

            densification_statistics_dict["dynamic_redundancy_threshold"] = dynamic_threshold
            densification_statistics_dict["static_redundancy_threshold"] = threshold
            densification_statistics_dict["dynamic_opacity_threshold"] = dynamic_opacity_threshold
            densification_statistics_dict["static_opacity_threshold"] = static_opacity_threshold

        self.prune_points(mask)
        densification_statistics_dict["n_points_mercied"] = mask.sum()
        densification_statistics_dict["redundancy_threshold"] = mean + lambda_mercy*std
        densification_statistics_dict["opacity_threshold"] = threshold if mercy_type in ['redundancy_opacity_opacity', 'opacity'] else 0

    def prune(self, max_grad, min_opacity, extent, max_screen_size, densification_statistics_dict, store_grads=False):
        n_points_pruned = super().prune(max_grad, min_opacity, extent, max_screen_size, store_grads)
        densification_statistics_dict["n_points_pruned"] = n_points_pruned

    def densify(self, max_grad, min_opacity, extent, max_screen_size, density_threshold, displacement_scale, model_path=None, iteration=None, stage=None, densification_statistics_dict=None, store_grads=False):
        n_points_cloned, n_points_split = super().densify(max_grad, min_opacity, extent, max_screen_size, density_threshold, displacement_scale, model_path, iteration, stage, store_grads)
        densification_statistics_dict["n_points_cloned"] = n_points_cloned
        densification_statistics_dict["n_points_split"] = n_points_split

