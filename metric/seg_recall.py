import einops
import torch
from torch.nn import Module
from torchmetrics.classification import MulticlassRecall

class SegRecallMetric(Module):
    
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.metric = MulticlassRecall(num_classes=num_classes, average=None)

    def forward(self, preds, target):
        assert len(preds.shape) == 4 # [B, C, H, W]
        assert len(target.shape) == 3 # [B, H, W]

        B, C, H, W = preds.shape
        total_recall = torch.zeros(size=(B, self.num_classes)).float().to(preds)

        for b in range(B):

            pred_batch = preds[b]
            pred_target = target[b]
            pred_batch = einops.rearrange(pred_batch, 'c h w -> (h w) c')
            pred_target = einops.rearrange(pred_target, 'h w -> (h w)')
        
            total_recall[b] = self.metric.to(preds)(preds, target)
        
        return total_recall