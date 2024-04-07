import torch
from torch import nn
from torch.nn import functional as F

class LOE(nn.Module):

    def __init__(self, sample_size:int=50) -> None:
        super().__init__()
        self.sample_size = sample_size

    def forward(self, input_tensor, target_tensor):
        B, C, H, W = input_tensor.shape

        sample_ratio = (self.sample_size * 1.0 / max(H, W))
        sample_h = round(sample_ratio * H)
        sample_w = round(sample_ratio * W)

        sample_input_tensor = F.interpolate(input_tensor, size=(sample_h, sample_w), mode='bilinear')
        sample_target_tensor = F.interpolate(target_tensor, size=(sample_h, sample_w), mode='bilinear')

        loe = torch.zeros(size=(B,)).float().to(input_tensor)

        for i in range(sample_h):
            for j in range(sample_w):
                U_A = sample_input_tensor[:, :, i:i+1, j:j+1] >= sample_input_tensor
                U_B = sample_target_tensor[:, :, i:i+1, j:j+1] >= sample_target_tensor
                RD = U_A ^ U_B
                loe += RD.float().mean(dim=(1, 2, 3))
        
        return loe / (sample_h * sample_w)
