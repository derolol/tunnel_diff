from typing import Any
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class SSIM(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor, target_tensor) -> Any:
        
        B, C, H, W = input_tensor.shape

        total_ssim = torch.zeros(size=(B,)).float().to(input_tensor)
        
        for c in range(C):
            input_c = input_tensor[:, c : c + 1, :, :]
            target_c = target_tensor[:, c : c + 1, :, :]
            total_ssim += self.ssim_single(input_c, target_c)

        return total_ssim / C

    def ssim_single(self, input_tensor, target_tensor):
        
        B, C, H, W = input_tensor.shape
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        kernel = torch.ones(1, 1, 8, 8) / 64
        kernel = kernel.to(input_tensor.device)

        mu_i = F.conv2d(input_tensor, kernel)
        mu_t = F.conv2d(target_tensor, kernel)

        var_i = F.conv2d(input_tensor ** 2, kernel) - mu_i ** 2
        var_t = F.conv2d(target_tensor ** 2, kernel) - mu_t ** 2
        cov_it = F.conv2d(target_tensor * input_tensor, kernel) - mu_i * mu_t

        a = (2 * mu_i * mu_t + C1) * (2 * cov_it + C2)
        b = (mu_i ** 2 + mu_t ** 2 + C1) * (var_i + var_t + C2)
        ssim_blocks = a / b
        
        return ssim_blocks.view(B, -1).mean(dim=1)

