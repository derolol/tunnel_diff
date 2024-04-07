import torch
from torch.nn import Module

class SegIoUMetric(Module):
    
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, preds, target):
        assert len(preds.shape) == 4 # [B, C, H, W]
        assert len(target.shape) == 3 # [B, H, W]
        
        preds = preds.argmax(dim=1)
        B, H, W = preds.shape

        total_iou = torch.zeros(size=(B, self.num_classes), dtype=torch.float).to(preds.device)

        for b in range(B):
            # 计算hist矩阵
            hist = self.get_hist(preds[b].flatten(), target[b].flatten(), self.num_classes)  
            
            # 计算各类别的IoU
            insert = hist.diag()
            union = hist.sum(dim=1) + hist.sum(dim=0) - hist.diag()
            union = union.maximum(torch.ones_like(hist.diag()))
            total_iou[b] = insert.float() / union.float()

        return total_iou

    def get_hist(self, pred, label, n):
        '''
        获取混淆矩阵
        label 标签 一维数组 HxW
        pred 预测结果 一维数组 HxW
        '''
        k = (label >= 0) & (label < n)
        # 对角线上的为分类正确的像素点
        return torch.bincount(n * label[k] + pred[k], minlength=n ** 2).reshape((n, n))