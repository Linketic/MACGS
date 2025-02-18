import imageio
import numpy as np
import torch
from scene import SceneMACGS
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
# from gaussian_renderer import GaussianModel
from scene.gaussian_model_macgs import GaussianModelMACGS
from time import time
import threading
import concurrent.futures

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, deformation, dynamic=False, model="quantised_half"):
    folder_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, deformation, model))
    render_path = os.path.join(folder_path, "renders")
    gts_path = os.path.join(folder_path, "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    gt_list = []
    render_list = []

    if dynamic:
        render_static_path = os.path.join(folder_path, "renders_static")
        render_dynamic_path = os.path.join(folder_path, "renders_dynamic")
        makedirs(render_static_path, exist_ok=True)
        makedirs(render_dynamic_path, exist_ok=True)
        render_static_list = []
        render_dynamic_list = []

    print("point nums:",gaussians._xyz.shape[0])

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        
        render_pkg = render(view, gaussians, pipeline, background, cam_type=cam_type, use_deformation=deformation, split_dynamic=dynamic, return_extra=True)
        rendering = render_pkg["render"]
        render_list.append(rendering)
        if dynamic:
            render_static_list.append(render_pkg["render_static"])
            render_dynamic_list.append(render_pkg["render_dynamic"])
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    imageio.mimwrite(os.path.join(folder_path, 'video_rgb.mp4'), [to8b(x).transpose(1, 2, 0) for x in render_list], fps=30)

    if len(os.listdir(gts_path)) == 0:
        multithread_write(gt_list, gts_path)
    multithread_write(render_list, render_path)

    if dynamic:
        multithread_write(render_static_list, render_static_path)
        multithread_write(render_dynamic_list, render_dynamic_path)
        imageio.mimwrite(os.path.join(folder_path, 'video_static.mp4'), [to8b(x).transpose(1, 2, 0) for x in render_static_list], fps=30)
        imageio.mimwrite(os.path.join(folder_path, 'video_dynamic.mp4'), [to8b(x).transpose(1, 2, 0) for x in render_dynamic_list], fps=30)

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, deformation="student", dynamic=False, model="quantised_half"):
    with torch.no_grad():
        gaussians = GaussianModelMACGS(dataset.sh_degree, hyperparam)
        scene = SceneMACGS(dataset, gaussians, load_iteration=iteration, shuffle=False, model=model)
        if deformation == "student":
            gaussians._student_deformation.load_state_dict(torch.load(os.path.join(dataset.model_path, "point_cloud", "iteration_14000", "distill_models", "student_deformation_best.pth")), strict=False)
            # gaussians._student_deformation.load_state_dict(torch.load(os.path.join(dataset.model_path, "point_cloud", "iteration_14000", "distill_models", "student_deformation_best_bak.pth")))
            # gaussians._student_deformation.load_state_dict(torch.load(os.path.join(dataset.model_path, "point_cloud", "iteration_14000", "student_deformation.pth")), strict=False)
            # gaussians._student_deformation.load_state_dict(torch.load(os.path.join(dataset.model_path, "point_cloud", "iteration_14000", "progressive_models", "deformation_7000.pth")))
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, cam_type, deformation, dynamic, model)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, cam_type, deformation, dynamic, model)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background, cam_type, deformation, dynamic, model)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", "-st", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", "-sv", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--deformation", "-d", type=str, default="student")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--model", type=str, default="quantised_half", choices=["baseline", "quantised", "quantised_half"])
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.deformation, args.dynamic, args.model)