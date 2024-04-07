from typing import Any
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class PSNR(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor, target_tensor) -> Any:
        B, C, H, W = input_tensor.shape
        mse = F.mse_loss(input_tensor, target_tensor, reduction='none')
        mse = mse.contiguous().view(B, -1).mean(dim=1)
        return 10 * torch.log10(1 / mse)

class PSNRB(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor, target_tensor) -> Any:

        B, C, H, W = input_tensor.shape

        total_psnr_b = torch.zeros(size=(B,)).float().to(input_tensor)
        
        for c in range(C):
            input_c = input_tensor[:, c : c + 1, :, :]
            target_c = target_tensor[:, c : c + 1, :, :]
            
            mse = F.mse_loss(input_c, target_c, reduction='none')
            bef = self.blocking_effect_factor(input_c)

            mse = mse.view(B, -1).mean(dim=1)
            total_psnr_b += 10 * torch.log10(1 / (mse + bef))

        return total_psnr_b / C

    def blocking_effect_factor(self, im):
        B, C, H, W = im.shape
        block_size = 8

        block_horizontal_positions = torch.arange(7, W - 1, 8)
        block_vertical_positions = torch.arange(7, H - 1, 8)

        horizontal_block_difference = ((im[:, :, :, block_horizontal_positions] - im[:, :, :, block_horizontal_positions + 1])**2).sum(3).sum(2).sum(1)
        vertical_block_difference = ((im[:, :, block_vertical_positions, :] - im[:, :, block_vertical_positions + 1, :])**2).sum(3).sum(2).sum(1)

        nonblock_horizontal_positions = np.setdiff1d(torch.arange(0, W - 1), block_horizontal_positions)
        nonblock_vertical_positions = np.setdiff1d(torch.arange(0, H - 1), block_vertical_positions)

        horizontal_nonblock_difference = ((im[:, :, :, nonblock_horizontal_positions] - im[:, :, :, nonblock_horizontal_positions + 1])**2).sum(3).sum(2).sum(1)
        vertical_nonblock_difference = ((im[:, :, nonblock_vertical_positions, :] - im[:, :, nonblock_vertical_positions + 1, :])**2).sum(3).sum(2).sum(1)

        n_boundary_horiz = H * (W // block_size - 1)
        n_boundary_vert = W * (H // block_size - 1)
        boundary_difference = (horizontal_block_difference + vertical_block_difference) / (n_boundary_horiz + n_boundary_vert)


        n_nonboundary_horiz = H * (W - 1) - n_boundary_horiz
        n_nonboundary_vert = W * (H - 1) - n_boundary_vert
        nonboundary_difference = (horizontal_nonblock_difference + vertical_nonblock_difference) / (n_nonboundary_horiz + n_nonboundary_vert)

        scaler = np.log2(block_size) / np.log2(min([H, W]))
        bef = scaler * (boundary_difference - nonboundary_difference)

        bef[boundary_difference <= nonboundary_difference] = 0
        
        return bef

