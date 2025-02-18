from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset


class FourDGSdataset(Dataset):
    def __init__(self, dataset: Neural3D_NDC_Dataset, args, dataset_type, load_image=True):
        self.dataset = dataset
        self.args = args
        self.dataset_type = dataset_type
        self.load_image = load_image

    def __getitem__(self, index):
        if self.dataset_type != "PanopticSports":
            try:
                if self.load_image:
                    image, w2c, time = self.dataset[index]
                    R, T = w2c
                    FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                    FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                    mask = None
                else:
                    image = None
                    w2c = self.dataset.load_pose(index)
                    time = self.dataset.load_time(index)
                    R, T = w2c
                    image_shape = self.dataset.load_image_shape()
                    FovX = focal2fov(self.dataset.focal[0], image_shape[1])
                    FovY = focal2fov(self.dataset.focal[0], image_shape[0])
                    mask = None
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                mask = caminfo.mask

            return Camera(
                colmap_id=index,
                R=R,
                T=T,
                FoVx=FovX,
                FoVy=FovY,
                image=image,
                gt_alpha_mask=None,
                image_name=f"{index}",
                uid=index,
                data_device=torch.device("cuda"),
                time=time,
                mask=mask
            )
        else:
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)