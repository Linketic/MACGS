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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time

def render(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color   = None, 
    stage            = "fine", 
    cam_type         = None, 
    iteration        = -1,
    split_dynamic    = False,
    use_deformation  = "teacher",  # "teacher" / "student",
    return_extra     = False
):
    """
    修改后的渲染函数,分别渲染静态和动态点
    """
    # 创建用于记录2D点的tensor
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 准备相机设置
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(pc.get_xyz.device).repeat(pc.get_xyz.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time = torch.tensor(viewpoint_camera['time']).to(pc.get_xyz.device).repeat(pc.get_xyz.shape[0],1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取基础数据
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # 准备scales和rotations 
    scales = pc._scaling
    rotations = pc._rotation
    deformation_point = pc._deformation_table if split_dynamic else torch.ones_like(pc._deformation_table, dtype=torch.bool, device=means3D.device) # true表示动态点, teacher_deformation会在全部点上进行估计, student_deformation只在动态点上进行估计
    static_mask = ~deformation_point  # 静态点的mask

    # 初始化最终变量
    means3D_final = means3D.clone()
    scales_final = scales.clone()
    rotations_final = rotations.clone()
    opacity_final = opacity.clone()
    shs_final = shs.clone()

    # 选择 deformation 网络
    if use_deformation == "teacher":
        deformation_net = pc._deformation
    elif use_deformation == "student":
        deformation_net = pc._student_deformation
    else:
        deformation_net = None

    # 处理动态点的变形
    if stage == "fine":
        means3D_deform, scales_deform, rotations_deform, opacity_deform, shs_deform = deformation_net(
            means3D[deformation_point], scales[deformation_point],
            rotations[deformation_point], opacity[deformation_point],
            shs[deformation_point], time[deformation_point]
        )
        
        means3D_final[deformation_point] = means3D_deform
        scales_final[deformation_point] = scales_deform
        rotations_final[deformation_point] = rotations_deform
        opacity_final[deformation_point] = opacity_deform
        shs_final[deformation_point] = shs_deform

        if iteration > 0:
            pc.update_deformation_accum(means3D_final, scales_final, rotations_final, opacity_final)
            if iteration % pc.deformation_update_interval == 0 and iteration > 3000:
                pc.update_deformation_table()

    # 激活函数处理
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)

    # 处理颜色
    colors_precomp = None
    if override_color is None and pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    elif override_color is not None:
        colors_precomp = override_color

    # 渲染完整图像
    full_image, full_radii, full_depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final)

    static_image = static_radii = static_depth = None
    dynamic_image = dynamic_radii = dynamic_depth = None
    if split_dynamic and return_extra: # 在训练时跳过
        static_image, static_radii, static_depth = rasterizer(
            means3D = means3D_final[static_mask],
            means2D = means2D[static_mask],
            shs = shs_final[static_mask],
            colors_precomp = colors_precomp[static_mask] if colors_precomp is not None else None,
            opacities = opacity_final[static_mask],
            scales = scales_final[static_mask],
            rotations = rotations_final[static_mask])

        dynamic_image, dynamic_radii, dynamic_depth = rasterizer(
            means3D = means3D_final[deformation_point],
            means2D = means2D[deformation_point],
            shs = shs_final[deformation_point], 
            colors_precomp = colors_precomp[deformation_point] if colors_precomp is not None else None,
            opacities = opacity_final[deformation_point],
            scales = scales_final[deformation_point],
            rotations = rotations_final[deformation_point])

    results =  {
        "render": full_image,
        "render_static": static_image,
        "render_dynamic": dynamic_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": full_radii > 0,
        "radii": full_radii,
        "depth": full_depth
    }

    return results