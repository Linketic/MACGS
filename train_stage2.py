import os
import sys
import random
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from scene import SceneMACGS, GaussianModelMACGS
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from gaussian_renderer import render, network_gui
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.timer import Timer
from utils.system_utils import mkdir_p
from collections import defaultdict
import imageio

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def collate_cameras(batch):
    """Custom collate function to keep camera objects unchanged"""
    return batch

class Trainer:
    def __init__(self, expname, dataset_args: ModelParams, optimization_args: OptimizationParams, 
                 pipeline_args: PipelineParams, model_args: ModelHiddenParams):
        self.expname = expname
        self.dataset_args = dataset_args
        self.opt_args = optimization_args
        self.opt_args.lambda_dssim = 0.01
        self.pipeline_args = pipeline_args
        self.model_args = model_args
        self.bg_color = torch.tensor([1,1,1], dtype=torch.float32, device="cuda") if self.dataset_args.white_background else torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

        # Load model and scene
        self.gaussians = GaussianModelMACGS(dataset_args.sh_degree, model_args)
        self.scene = SceneMACGS(dataset_args, self.gaussians, load_iteration=-1)
        self.gaussians.initialize_student_network()
        self.save_path = os.path.join(self.dataset_args.model_path, "point_cloud", f"iteration_{self.scene.loaded_iter}")
        
        # Freeze model parameters
        for param in self.gaussians._deformation.parameters():
            param.requires_grad = False
        
        # Freeze gaussian point parameters
        self.gaussians._xyz.requires_grad = False
        self.gaussians._features_dc.requires_grad = False
        self.gaussians._features_rest.requires_grad = False
        self.gaussians._scaling.requires_grad = False
        self.gaussians._rotation.requires_grad = False
        self.gaussians._opacity.requires_grad = False
        
        self._setup_optimizer()

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        if os.path.exists(checkpoint_path):
            model_params, _ = torch.load(checkpoint_path)
            self.gaussians.restore(model_params, self.opt_args)
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    def _setup_optimizer(self):
        """Setup optimizer - only optimize student_deformation network parameters"""
        param_groups = []
        
        mlp_params = list(self.gaussians._student_deformation.get_mlp_parameters())
        if mlp_params:
            param_groups.append({
                'params': mlp_params,
                'lr': self.opt_args.deformation_lr_init * self.opt_args.student_lr_multiplier,
                'name': 'student_deformation'
            })

        grid_params = list(self.gaussians._student_deformation.get_grid_parameters())
        if grid_params:
            param_groups.append({
                'params': grid_params,
                'lr': self.opt_args.grid_lr_init * self.opt_args.student_lr_multiplier,
                'name': 'student_grid'
            })

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        """Update learning rate"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "student_grid":
                param_group['lr'] = self._get_expon_lr(
                    iteration,
                    self.opt_args.grid_lr_init * self.opt_args.student_lr_multiplier,
                    self.opt_args.grid_lr_final,
                    self.opt_args.deformation_lr_delay_mult * self.opt_args.student_lr_decay_multiplier
                )
            else:
                param_group['lr'] = self._get_expon_lr(
                    iteration,
                    self.opt_args.deformation_lr_init * self.opt_args.student_lr_multiplier,
                    self.opt_args.deformation_lr_final,
                    self.opt_args.deformation_lr_delay_mult * self.opt_args.student_lr_decay_multiplier
                )

    def _get_expon_lr(self, iter, lr_init, lr_final, lr_delay_mult):
        """Calculate exponentially decaying learning rate"""
        if iter < 0:
            return lr_init
        if lr_delay_mult == 0:
            return lr_final
        alpha = iter / (self.opt_args.iterations / lr_delay_mult)
        return lr_final + (lr_init - lr_final) * (1 - min(1, alpha))

    def evaluate_model(self, cameras, save_video=False):
        """Evaluate model performance"""
        total_psnr = 0.0
        total_loss = 0.0
        renders = []
        
        with torch.no_grad():
            for camera in cameras:
                student_output = render(
                    camera,
                    self.gaussians,
                    self.pipeline_args,
                    self.bg_color,
                    stage="fine",
                    use_deformation="student",
                    split_dynamic=True
                )

                gt_image = camera.original_image.to("cuda")
                
                psnr_value = psnr(student_output["render"], gt_image).mean().double().item()
                loss = l1_loss(student_output["render"], gt_image).item()
                
                total_psnr += psnr_value
                total_loss += loss

                renders.append(student_output["render"])
        
        if save_video:
            imageio.mimwrite(os.path.join(self.save_path, "distill_video.mp4"), [to8b(x).transpose(1, 2, 0) for x in renders], fps=30)
                
        return total_psnr / len(cameras), total_loss / len(cameras), student_output["render"]

    def train_step(self, viewpoint_cams):
        """Execute one step of distillation training"""
        self.optimizer.zero_grad()
        total_loss = 0
        metrics = {
            'l1_loss': 0,
            'ssim_loss': 0,
            'tv_loss': 0,
            'gt_loss': 0,
            'psnr': 0
        }
        
        for viewpoint_cam in viewpoint_cams:
            student_out = render(
                viewpoint_cam,
                self.gaussians, 
                self.pipeline_args,
                self.bg_color,
                stage="fine",
                use_deformation="student",
                split_dynamic=True,
            )

            gt_image = viewpoint_cam.original_image.to("cuda")
            
            gt_l1_loss = l1_loss(student_out["render"], gt_image)
            metrics['gt_loss'] += gt_l1_loss.item()

            if self.opt_args.lambda_dssim != 0:
                gt_ssim_loss = 1.0 - ssim(student_out["render"], gt_image)
                metrics['ssim_loss'] += gt_ssim_loss.item()
                gt_l1_loss += self.opt_args.lambda_dssim * gt_ssim_loss
            
            if self.model_args.time_smoothness_weight != 0:
                tv_loss = self.gaussians.compute_regulation(self.model_args.time_smoothness_weight, self.model_args.l1_time_planes, self.model_args.plane_tv_weight, is_student=True)
                metrics['tv_loss'] += tv_loss.item()
            else:
                tv_loss = torch.tensor([0.], device="cuda")

            metrics['psnr'] += psnr(student_out["render"], gt_image).mean().double().item()

            loss = gt_l1_loss + tv_loss
            total_loss += loss
        
        total_loss /= len(viewpoint_cams)
        for key in metrics:
            metrics[key] /= len(viewpoint_cams)
            
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.gaussians._student_deformation.parameters(),
            max_norm=1.0
        )
        
        return total_loss.item(), metrics

    def save_model(self, iteration, best=False, save_scene=False):
        """Save model"""
        models_dir = os.path.join(self.save_path, "distill_models")
        mkdir_p(models_dir)

        model_name = "student_deformation_best.pth" if best else f"student_deformation_{iteration}.pth"
        model_path = os.path.join(models_dir, model_name)

        try:
            torch.save(self.gaussians._student_deformation.state_dict(), model_path)
            print(f"Model saved at {model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
        
        if save_scene:
            self.scene.save(iteration, "distilled")

    def train(self, num_iterations):
        """Execute distillation training"""
        tb_writer = None
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(self.dataset_args.model_path)

        progress_bar = tqdm(range(num_iterations), desc="Distillation Training", mininterval=10.0, miniters=10)
        
        train_cameras = self.scene.getTrainCameras()
        test_cameras = self.scene.getTestCameras()
        
        train_loader = DataLoader(
            train_cameras, 
            batch_size=self.opt_args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_cameras
        )
        train_iter = iter(train_loader)
        
        best_psnr = 0
        
        for iteration in range(num_iterations):
            self.update_learning_rate(iteration)
            
            try:
                viewpoint_cams = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                viewpoint_cams = next(train_iter)
            
            loss, metrics = self.train_step(viewpoint_cams)
            
            self.optimizer.step()
            
            if tb_writer:
                tb_writer.add_scalar('distill/total_loss', loss, iteration)
                for key, value in metrics.items():
                    tb_writer.add_scalar(f'distill/{key}', value, iteration)
            
            if (iteration + 1) % 1000 == 0 and (iteration + 1) > 4000:
                psnr_value, test_loss, student_render = self.evaluate_model(test_cameras)
                print(f"\nIteration {iteration+1}")
                print(f"Test PSNR: {psnr_value:.2f}")
                print(f"Test Loss: {test_loss:.6f}")
                
                if tb_writer:
                    tb_writer.add_scalar('distill_test/psnr', psnr_value, iteration)
                    tb_writer.add_scalar('distill_test/loss', test_loss, iteration)
                    tb_writer.add_image('distill_test/render', student_render, iteration)
                
                self.save_model(iteration + 1)
                
                if psnr_value > best_psnr:
                    best_psnr = psnr_value
                    self.save_model(iteration + 1, best=True)
                    print(f"New best model saved! PSNR: {best_psnr:.2f}")
            
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "L": f"{loss:.7f}",
                    "GT L": f"{metrics['gt_loss']:.7f}",
                    "TV L": f"{metrics['tv_loss']:.7f}",
                    "SS L": f"{metrics['ssim_loss']:.7f}",
                    "PSNR": f"{metrics['psnr']:.2f}"
                })
                progress_bar.update(10)

        progress_bar.close()
        if tb_writer:
            tb_writer.close()
        
        print(f"\nTraining completed. Best PSNR: {best_psnr:.2f}")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = ArgumentParser(description="Training Script")
    parser.add_argument("--num_iterations", type=int, default=7000,
                      help="Number of distillation iterations")
    parser.add_argument("--expname", type=str, default="",
                      help="Experiment name")
    parser.add_argument("--configs", type=str, default="",
                      help="Path to config file")
    parser.add_argument('--ip', type=str, default="127.0.0.1",
                      help="IP for GUI server")
    parser.add_argument('--port', type=int, default=6009,
                      help="Port for GUI server")
    parser.add_argument('--debug_from', type=int, default=-1,
                      help="Debug from iteration number")
    parser.add_argument('--detect_anomaly', action='store_true', default=False,
                      help="Enable PyTorch anomaly detection")
    
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    args = parser.parse_args()
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    # 设置输出路径
    if not args.model_path:
        if args.expname:
            unique_str = args.expname
        else:
            import uuid
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
    
    print("Output folder:", args.model_path)
    os.makedirs(args.model_path, exist_ok=True)
    
    with open(os.path.join(args.model_path, "cfg_args_2"), 'w') as f:
        f.write(str(args))
    
    setup_seed(6666)

    while True:
        try:
            network_gui.init(args.ip, args.port)
            break
        except:
            args.port += 1
            
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    trainer = Trainer(
        unique_str,
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        hp.extract(args),
    )
    
    trainer.train(args.num_iterations)
    
    print(f"max cuda memory allocated: {torch.cuda.max_memory_allocated('cuda') / 1024**2} MB")
    print("\nDistillation training complete.")

if __name__ == "__main__":
    main()
