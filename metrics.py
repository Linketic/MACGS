from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim
from natsort import natsorted
import concurrent.futures
from typing import Tuple, List, Dict, Set


def read_single_image(args: Tuple[Path, Path, str]) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """读取单张图像对的辅助函数"""
    renders_dir, gt_dir, fname = args
    render = Image.open(renders_dir / fname)
    gt = Image.open(gt_dir / fname)
    render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
    gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
    return render_tensor, gt_tensor, fname


def readImages(renders_dir: Path, gt_dir: Path) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """多线程读取图像"""
    renders, gts, image_names = [], [], []
    
    # 准备参数列表
    args_list = [(renders_dir, gt_dir, fname) for fname in os.listdir(renders_dir)]
    
    # 使用线程池读取图像
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = executor.map(read_single_image, args_list)
        
    # 整理结果
    for render_tensor, gt_tensor, fname in results:
        renders.append(render_tensor)
        gts.append(gt_tensor)
        image_names.append(fname)
    
    return renders, gts, image_names


def get_available_metrics() -> Dict[str, callable]:
    """返回所有可用的指标及其计算函数"""
    return {
        "ssim": lambda x, y: ssim(x, y),
        "psnr": lambda x, y: psnr(x, y),
        "lpips_vgg": lambda x, y: lpips(x, y, net_type='vgg'),
        "lpips_alex": lambda x, y: lpips(x, y, net_type='alex'),
        "ms_ssim": lambda x, y: ms_ssim(x, y, data_range=1, size_average=True),
        "d_ssim": lambda x, y: (1 - ms_ssim(x, y, data_range=1, size_average=True)) / 2
    }


def evaluate(model_paths: List[str], selected_metrics: Set[str]):
    available_metrics = get_available_metrics()
    
    # 验证选择的指标是否有效
    invalid_metrics = selected_metrics - set(available_metrics.keys())
    if invalid_metrics:
        raise ValueError(f"Invalid metrics selected: {invalid_metrics}")
    
    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in natsorted(os.listdir(test_dir)):
                print("Method:", method)

                # Initialize dictionaries for this method
                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                # Setup directories
                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                # Calculate selected metrics
                results = {metric: [] for metric in selected_metrics}

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    for metric in selected_metrics:
                        results[metric].append(
                            available_metrics[metric](renders[idx], gts[idx])
                        )

                # Print results
                for metric in selected_metrics:
                    mean_value = torch.tensor(results[metric]).mean()
                    print(f"Scene: {scene_dir} {metric:10}: {mean_value:>12.7f}")
                    
                    # Update dictionaries
                    full_dict[scene_dir][method][metric] = mean_value.item()
                    per_view_dict[scene_dir][method][metric] = {
                        name: value for value, name in zip(
                            torch.tensor(results[metric]).tolist(),
                            image_names
                        )
                    }

            # Save results
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)

        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            raise e


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    
    # 添加指标选择参数
    available_metrics = get_available_metrics().keys()
    parser.add_argument(
        '--metrics',
        nargs="+",
        type=str,
        default=list(available_metrics),  # 默认计算所有指标
        # default=["ssim", "psnr"],
        choices=available_metrics,
        help=f"Select metrics to compute. Available metrics: {', '.join(available_metrics)}"
    )
    
    args = parser.parse_args()
    evaluate(args.model_paths, set(args.metrics))