import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0
def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class HexPlaneField(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)
    
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)


        return features # [pts.shape[0], self.feat_dim*len(self.multiscale_res_multipliers)]

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features
    
    def initialize_student(self, teacher_grid, ratio=0.5):
        """使用teacher的grid参数初始化student"""
        student_grids = downsample_grid_params(teacher_grid, ratio)
        self.grids = student_grids
        return self
    
    def progressive_downsample_student(self, teacher_grid, ratio_x=1.0, ratio_y=1.0, ratio_z=1.0):
        """渐进式下采样teacher的grid参数"""
        student_grids = progressive_downsample_grid_params(teacher_grid, ratio_x, ratio_y, ratio_z)
        self.grids = student_grids
        return self


def downsample_grid_params(teacher_grid: HexPlaneField, ratio=0.5):
    """下采样teacher的grid参数来初始化student
    
    对于6个平面:
    - idx 0,1,3: 空间平面 [1, out_dim, H, W] -> [1, out_dim, H*ratio, W*ratio]
    - idx 2,4,5: 时空平面 [1, out_dim, T, W] -> [1, out_dim, T, W*ratio]
    """
    student_grids = nn.ModuleList()
    
    for grid in teacher_grid.grids:
        student_grid_coefs = nn.ParameterList()
        for idx, coef in enumerate(grid):
            shape = list(coef.shape)
            
            if idx in [2, 4, 5]:  # 时空平面：只减小最后一维(空间维度)
                new_spatial_dims = [
                    shape[2],  # 保持时间维度不变
                    int(shape[3] * ratio)  # 减小空间维度
                ]
            else:  # 空间平面：同时减小两个维度
                new_spatial_dims = [int(s * ratio) for s in shape[2:]]
            
            # 使用插值下采样
            if len(new_spatial_dims) > 0:
                downsampled = F.interpolate(
                    coef,
                    size=new_spatial_dims,
                    mode='bilinear',  # 都是2D平面，使用bilinear插值
                    align_corners=True
                )
                student_grid_coefs.append(nn.Parameter(downsampled))
            else:
                student_grid_coefs.append(nn.Parameter(coef.clone()))
                
        student_grids.append(student_grid_coefs)
    
    return student_grids

def progressive_downsample_grid_params(teacher_grid: HexPlaneField, ratio_x=1.0, ratio_y=1.0, ratio_z=1.0):
    """渐进式下采样teacher的grid参数
    
    Args:
        teacher_grid: 教师模型的HexPlaneField
        ratio_x: x维度的下采样比例 (0.0-1.0)
        ratio_y: y维度的下采样比例 (0.0-1.0)
        ratio_z: z维度的下采样比例 (0.0-1.0)
    
    Returns:
        student_grids: 下采样后的grid参数
    """
    student_grids = nn.ModuleList()

    for grid in teacher_grid.grids:
        student_grid_coefs = nn.ParameterList()
        
        # 遍历6个平面
        for idx, coef in enumerate(grid):
            shape = list(coef.shape)
            
            if idx in [2, 4, 5]:  # 时空平面
                # idx 2: T-X plane
                # idx 4: T-Y plane
                # idx 5: T-Z plane
                spatial_dim = idx - 2  # 0:X, 1:Y, 2:Z
                ratio = [1.0, ratio_x if spatial_dim == 0 else ratio_y if spatial_dim == 1 else ratio_z]
                new_spatial_dims = [
                    shape[2],  # 保持时间维度不变
                    int(shape[3] * ratio[1])  # 仅缩放空间维度
                ]
            else:  # 空间平面
                # idx 0: X-Y plane
                # idx 1: Y-Z plane
                # idx 3: X-Z plane
                if idx == 0:      # X-Y plane
                    ratio = [ratio_x, ratio_y]
                elif idx == 1:    # Y-Z plane
                    ratio = [ratio_y, ratio_z]
                else:            # X-Z plane (idx == 3)
                    ratio = [ratio_x, ratio_z]
                    
                new_spatial_dims = [
                    int(shape[2] * ratio[0]),
                    int(shape[3] * ratio[1])
                ]
            
            # 使用双线性插值进行下采样
            if len(new_spatial_dims) > 0:
                downsampled = F.interpolate(
                    coef,
                    size=new_spatial_dims,
                    mode='bilinear',
                    align_corners=True
                )
                student_grid_coefs.append(nn.Parameter(downsampled))
            else:
                student_grid_coefs.append(nn.Parameter(coef.clone()))
                
        student_grids.append(student_grid_coefs)
    
    return student_grids
