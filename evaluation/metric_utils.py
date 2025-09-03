# This code was adapted from EasyVolcap: https://github.com/zju3dv/EasyVolcap/blob/main/easyvolcap/utils/metric_utils.py
"""
Given images, output scalar metrics on CPU
Used for evaluation. For training, please check out loss_utils
"""

import torch
import numpy as np
from PIL import Image

# from easyvolcap.utils.console_utils import *
# from easyvolcap.utils.loss_utils import mse as compute_mse
# from easyvolcap.utils.loss_utils import lpips as compute_lpips
from skimage.metrics import structural_similarity as compare_ssim

from enum import Enum, auto
import lpips

def compute_mse(x: torch.Tensor, y: torch.Tensor):
    return ((x.float() - y.float())**2).mean()
    
class LPIPSModelSingleton:
    _alex_model = None
    _vgg_model = None

    @staticmethod
    def get_alex_model():
        if LPIPSModelSingleton._alex_model is None:
            LPIPSModelSingleton._alex_model = lpips.LPIPS(net='alex', verbose=False).cuda()
        return LPIPSModelSingleton._alex_model

    @staticmethod
    def get_vgg_model():
        if LPIPSModelSingleton._vgg_model is None:
            LPIPSModelSingleton._vgg_model = lpips.LPIPS(net='vgg', verbose=False).cuda()
        return LPIPSModelSingleton._vgg_model

def compute_lpips_alex(x: torch.Tensor, y: torch.Tensor):
    lpips_model = LPIPSModelSingleton.get_alex_model()
    return lpips_model(x.cuda() * 2 - 1, y.cuda() * 2 - 1).mean()

def compute_lpips_vgg(x: torch.Tensor, y: torch.Tensor):
    lpips_model = LPIPSModelSingleton.get_vgg_model()
    return lpips_model(x.cuda() * 2 - 1, y.cuda() * 2 - 1).mean()

@torch.no_grad()
def psnr(x: torch.Tensor, y: torch.Tensor):
    mse = compute_mse(x, y).mean()
    psnr = (1 / mse.clip(1e-10)).log() * 10 / np.log(10)
    return psnr.item()  # tensor to scalar


@torch.no_grad()
def ssim(x: torch.Tensor, y: torch.Tensor):
    return np.mean([
        compare_ssim(
            _x.detach().cpu().numpy(),
            _y.detach().cpu().numpy(),
            channel_axis=-1,
            data_range=1.0, #2.0
        )
        for _x, _y in zip(x, y)
    ]).astype(float).item()


@torch.no_grad()
def lpips_alex(x: torch.Tensor, y: torch.Tensor):
    if x.ndim == 3: x = x.unsqueeze(0)
    if y.ndim == 3: y = y.unsqueeze(0)
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    return compute_lpips_alex(x, y).item()

@torch.no_grad()
def lpips_vgg(x: torch.Tensor, y: torch.Tensor):
    if x.ndim == 3: x = x.unsqueeze(0)
    if y.ndim == 3: y = y.unsqueeze(0)
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    return compute_lpips_vgg(x, y).item()


class Metrics(Enum):
    PSNR = psnr
    SSIM = ssim
    LPIPS_ALEX = lpips_alex
    LPIPS_VGG = lpips_vgg
