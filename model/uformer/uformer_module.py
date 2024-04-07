import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR, StepLR

from pytorch_lightning import LightningModule

from metric.loe import LOE
from metric.niqe.niqe import NIQE
from metric.psnr import PSNR, PSNRB
from metric.ssim import SSIM

from model.base_module import BaseModelModule
from model.uformer.uformer import Uformer

class UFormerModule(BaseModelModule):

    def __init__(self,
                 arch:str='Uformer_B',
                 dd_in:int=3,
                 train_ps:int=128,
                 lr:float=0.0002,
                 optimizer:str='adamw',
                 warmup:bool=True,
                 weight_decay:float=0.02):
        
        super().__init__()
        
        self.arch = arch
        self.dd_in = dd_in
        self.train_ps = train_ps
        self.lr = lr
        self.optimizer = optimizer
        self.warmup = warmup
        self.weight_decay = weight_decay
        
        if self.arch == 'Uformer_B':
            self.model = Uformer(img_size=self.train_ps,
                                 embed_dim=32,
                                 win_size=8,
                                 token_projection='linear',
                                 token_mlp='leff',
                                 depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
                                 modulator=True,
                                 dd_in=self.dd_in)  
        else:
            raise Exception("Arch error!")

        self.loss = CharbonnierLoss()

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

        input_restored = self.model(input_low)

        loss = self.loss(input_restored, input_high)
        
        return loss, input_restored

    def training_step(self, batch, batch_idx):

        loss, input_restored = self(batch)

        self.log("train/loss", loss)

        return {"loss": loss,
                "output": input_restored}
    
    def validation_step(self, batch, batch_idx):

        input_low = batch["hint"]
        input_high = batch["jpg"]

        loss, input_restored = self(batch)

        metric_loe = self.metric_loe(input_restored, input_high).mean()
        metric_niqe = self.metric_niqe(input_restored).mean()
        metric_psnr = self.metric_psnr(input_restored, input_high).mean()
        metric_psnr_b = self.metric_psnr_b(input_restored, input_high).mean()
        metric_ssim = self.metric_ssim(input_restored, input_high).mean()

        self.log(name="val/metric_loe", value=metric_loe)
        self.log(name="val/metric_niqe", value=metric_niqe)
        self.log(name="val/metric_psnr", value=metric_psnr)
        self.log(name="val/metric_psnr_b", value=metric_psnr_b)
        self.log(name="val/metric_ssim", value=metric_ssim)

        return input_restored
    
    def forward_enhance(self, batch, batch_idx):
        input_low = batch["hint"]
        input_enhance = self.model(input_low)
        return input_enhance

    def configure_optimizers(self):

        # Optmizer

        if self.optimizer.lower() == 'adam':
            optimizer = Adam(self.model.parameters(),
                             lr=self.lr,
                             betas=(0.9, 0.999),
                             eps=1e-8,
                             weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adamw':
            optimizer = AdamW(self.model.parameters(),
                              lr=self.lr,
                              betas=(0.9, 0.999),
                              eps=1e-8,
                              weight_decay=self.weight_decay)
        else:
            raise Exception("Error optimizer...")
        
        # Scheduler
        
        if self.warmup:
            scheduler = OneCycleLR(optimizer=optimizer,
                                   max_lr=self.lr,
                                   total_steps=self.trainer.max_steps)
        else:
            raise Exception("Error scheduler...")

        return [optimizer], [scheduler]
    
    def log_images(self, batch, outputs, sample_steps=50):
        log = dict()

        input_low = batch["hint"]
        input_high = batch["jpg"]

        input_restored = outputs

        B, C, H, W = input_low.shape

        for b in range(min(2, B)):
            
            log[f"b{b}"] = torch.stack(
                [
                    (input_low[b] + 1) / 2, # lq
                    (input_high[b] + 1) / 2, # hq
                    (input_restored[b] + 1) / 2, # input_restored
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
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss