import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningModule

from metric.loe import LOE
from metric.niqe.niqe import NIQE
from metric.psnr import PSNR, PSNRB
from metric.ssim import SSIM

from model.base_module import BaseModelModule
from model.iat.iat import IAT

class IATModule(BaseModelModule):

    def __init__(self,
                 lr:float=0.0002,
                 weight_decay:float=0.0002):
        
        super().__init__()
        
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.model = IAT()

        self.l1_loss = nn.L1Loss()
        # self.l1_smooth_loss = nn.SmoothL1Loss()
        # self.vgg_model = vgg16(pretrained=True).features[:16]
        # for param in self.vgg_model.parameters():
        #     param.requires_grad = False
        # self.network_loss = LossNetwork(self.vgg_model)
        # self.network_loss.eval()

        self.metric_loe = LOE()
        self.metric_niqe = NIQE()
        self.metric_psnr = PSNR()
        self.metric_psnr_b = PSNRB()
        self.metric_ssim = SSIM()

    def log(self, name, value):
        super().log(name=name,
                    value=value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    logger=True)

    def forward(self, batch):

        input_low = batch["hint"]
        input_high = batch["jpg"]

        mul, add, input_enhance = self.model(input_low)
    
        loss = self.l1_loss(input_enhance, input_high)
        
        return loss, input_enhance

    def training_step(self, batch, batch_idx):

        loss, input_enhance = self(batch)

        self.log("train/loss", loss)

        return {"loss": loss,
                "output": input_enhance}
    
    def validation_step(self, batch, batch_idx):

        input_low = batch["hint"]
        input_high = batch["jpg"]

        loss, input_enhance = self(batch)

        metric_loe = self.metric_loe(input_enhance, input_high).mean()
        metric_niqe = self.metric_niqe(input_enhance).mean()
        metric_psnr = self.metric_psnr(input_enhance, input_high).mean()
        metric_psnr_b = self.metric_psnr_b(input_enhance, input_high).mean()
        metric_ssim = self.metric_ssim(input_enhance, input_high).mean()

        self.log(name="val/metric_loe", value=metric_loe)
        self.log(name="val/metric_niqe", value=metric_niqe)
        self.log(name="val/metric_psnr", value=metric_psnr)
        self.log(name="val/metric_psnr_b", value=metric_psnr_b)
        self.log(name="val/metric_ssim", value=metric_ssim)

        return input_enhance
    
    def forward_enhance(self, batch, batch_idx):
        input_low = batch["hint"]
        mul, add, input_enhance = self.model(input_low)
        return input_enhance

    def configure_optimizers(self):

        # Optmizer
        optimizer = Adam(self.model.parameters(),
                         lr=self.lr,
                         weight_decay=self.weight_decay)
        
        # Scheduler
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=self.trainer.max_steps)

        return [optimizer], [scheduler]
    
    def log_images(self, batch, outputs, sample_steps=50):
        log = dict()

        input_low = batch["hint"]
        input_high = batch["jpg"]

        input_enhance = outputs

        B, C, H, W = input_low.shape

        for b in range(min(2, B)):
            
            log[f"b{b}"] = torch.stack(
                [
                    (input_low[b] + 1) / 2, # lq
                    (input_high[b] + 1) / 2, # hq
                    (input_enhance[b] + 1) / 2, # input_enhance
                ],
                dim=0
            )

        return log

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

# Perpectual Loss
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)