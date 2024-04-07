import einops
import torch
from torch.nn import Module
from torchmetrics.classification import MulticlassFBetaScore

class SegFBetaScoreMetric(Module):
    
    def __init__(self, num_classes, beta=1.) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.metric = MulticlassFBetaScore(num_classes=num_classes, beta=beta, average=None)

    def forward(self, preds, target):
        assert len(preds.shape) == 4 # [B, C, H, W]
        assert len(target.shape) == 3 # [B, H, W]

        B, C, H, W = preds.shape
        total_fbeta_score = torch.zeros(size=(B, self.num_classes)).float().to(preds)

        for b in range(B):

            pred_batch = preds[b]
            pred_target = target[b]
            pred_batch = einops.rearrange(pred_batch, 'c h w -> (h w) c')
            pred_target = einops.rearrange(pred_target, 'h w -> (h w)')
        
            total_fbeta_score[b] = self.metric.to(preds)(preds, target)
        
        return total_fbeta_score