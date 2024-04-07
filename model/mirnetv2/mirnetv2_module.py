import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import L1Loss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler

from pytorch_lightning import LightningModule

from metric.loe import LOE
from metric.niqe.niqe import NIQE
from metric.psnr import PSNR, PSNRB
from metric.ssim import SSIM

from model.base_module import BaseModelModule
from model.mirnetv2.mirnetv2 import MIRNet_v2

class MIRNetV2Module(BaseModelModule):

    def __init__(self,
                 lr,
                 betas,
                 periods,
                 restart_weights,
                 eta_mins):
        
        super().__init__()
        
        self.lr = lr
        self.betas = betas
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins

        self.model = MIRNet_v2(
            inp_channels = 3,
            out_channels = 3,
            n_feat = 80,
            chan_factor = 1.5,
            n_RRG = 4,
            n_MRB = 2,
            height = 3,
            width = 2,
            scale = 1)

        

        self.loss = L1Loss()

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

        input_pred = self.model(input_low)
        loss = self.loss(input_pred, input_high)

        return loss, input_pred

    def training_step(self, batch, batch_idx):

        loss, input_pred = self(batch)

        self.log("train/loss", loss)

        return {"loss": loss,
                "output": input_pred}
    
    def validation_step(self, batch, batch_idx):

        input_low = batch["hint"]
        input_high = batch["jpg"]

        loss, input_pred = self(batch)

        metric_loe = self.metric_loe(input_pred, input_high).mean()
        metric_niqe = self.metric_niqe(input_pred).mean()
        metric_psnr = self.metric_psnr(input_pred, input_high).mean()
        metric_psnr_b = self.metric_psnr_b(input_pred, input_high).mean()
        metric_ssim = self.metric_ssim(input_pred, input_high).mean()

        self.log(name="val/metric_loe", value=metric_loe)
        self.log(name="val/metric_niqe", value=metric_niqe)
        self.log(name="val/metric_psnr", value=metric_psnr)
        self.log(name="val/metric_psnr_b", value=metric_psnr_b)
        self.log(name="val/metric_ssim", value=metric_ssim)

        return input_pred

    def forward_enhance(self, batch, batch_idx):
        input_low = batch["hint"]
        input_enhance = self.model(input_low)
        return input_enhance

    def configure_optimizers(self):
        optim_params = []

        for k, v in self.model.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                self.print(f'Params {k} will not be optimized.')

        # Optmizer
        optimizer = torch.optim.Adam(optim_params, lr=self.lr, betas=self.betas)
        
        # Scheduler
        scheduler = CosineAnnealingRestartCyclicLR(optimizer,
                                                   periods=self.periods,
                                                   restart_weights=self.restart_weights,
                                                   eta_mins=self.eta_mins)

        return [optimizer], [scheduler]
    
    def log_images(self, batch, outputs, sample_steps=50):
        log = dict()

        input_low = batch["hint"]
        input_high = batch["jpg"]

        input_pred = outputs

        B, C, H, W = input_low.shape

        for b in range(min(2, B)):
            
            log[f"b{b}"] = torch.stack(
                [
                    (input_low[b] + 1) / 2, # lq
                    (input_high[b] + 1) / 2, # hq
                    (input_pred[b] + 1) / 2, # input_pred
                ],
                dim=0
            )

        return log

def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i

class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.
    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7
    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.
    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=(1, ),
                 eta_mins=(0, ),
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]