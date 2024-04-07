import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from pytorch_lightning import LightningModule

from metric.loe import LOE
from metric.niqe.niqe import NIQE
from metric.psnr import PSNR, PSNRB
from metric.ssim import SSIM

from model.base_module import BaseModelModule
from model.zero_dce.zero_dce import DCENet

class ZeroDceModule(BaseModelModule):

    def __init__(self,
                 lr:float=1e-4,
                 wd:float=0,
                 lr_decay_factor:float=0.97,
                 loss:int=1,
                 w_spa:float=8.0,
                 w_exp:float=1.75,
                 w_col:float=1.0,
                 w_tvA:float=7.0,
                 spa_rsize:int=4,
                 exp_rsize:int=16):
        
        super().__init__()
        
        self.lr = lr
        self.weight_decay = wd
        self.lr_decay_factor = lr_decay_factor
        self.loss = loss
        self.w_spa = w_spa
        self.w_exp = w_exp
        self.w_col = w_col
        self.w_tvA = w_tvA
        self.spa_rsize = spa_rsize
        self.exp_rsize = exp_rsize

        to_gray, neigh_diff = self.get_kernels()
        self.to_gray = to_gray
        self.neigh_diff = neigh_diff

        self.alpha_n = 8
        self.return_result_index = [4, 6, 8]

        self.model = DCENet(n=self.alpha_n,
                            return_results=self.return_result_index)

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

        output = self.model(input_low)
        results, Astack = output
        input_enhanced = results[-1]

        L_spa = self.w_spa * self.spatial_consistency_loss(input_enhanced,
                                                           input_low,
                                                           self.to_gray.to(self.device),
                                                           self.neigh_diff.to(self.device),
                                                           self.spa_rsize)
        L_exp = self.w_exp * self.exposure_control_loss(input_enhanced,
                                                        self.exp_rsize,
                                                        E=0.62)
        if self.loss == 1:
            L_col = self.w_col * self.color_constency_loss(input_enhanced)
        elif self.loss == 2:
            L_col = self.w_col * self.color_constency_loss2(input_enhanced, input_low)
        
        L_tvA = self.w_tvA * self.alpha_total_variation(Astack)
        
        loss = L_spa + L_exp + L_col + L_tvA

        return loss, output

    def training_step(self, batch, batch_idx):

        loss, output = self(batch)
        self.log("train/loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):

        input_low = batch["hint"]
        input_high = batch["jpg"]

        loss, output = self(batch)
        self.log("val/loss", loss)

        results, Astack = output
        input_enhanced = results[-1]

        metric_loe = self.metric_loe(input_enhanced, input_high).mean()
        metric_niqe = self.metric_niqe(input_enhanced).mean()
        metric_psnr = self.metric_psnr(input_enhanced, input_high).mean()
        metric_psnr_b = self.metric_psnr_b(input_enhanced, input_high).mean()
        metric_ssim = self.metric_ssim(input_enhanced, input_high).mean()

        self.log(name="val/metric_loe", value=metric_loe)
        self.log(name="val/metric_niqe", value=metric_niqe)
        self.log(name="val/metric_psnr", value=metric_psnr)
        self.log(name="val/metric_psnr_b", value=metric_psnr_b)
        self.log(name="val/metric_ssim", value=metric_ssim)

        return input_low, input_high, results, Astack

    def forward_enhance(self, batch, batch_idx):
        input_low = batch["hint"]
        results, Astack = self.model(input_low)
        input_enhance = results[-1]
        return input_enhance

    def configure_optimizers(self):
        
        optimizer = Adam(self.model.parameters(),
                         lr=self.lr,
                         weight_decay=self.weight_decay)
        
        # scheduler = ReduceLROnPlateau(optimizer,
        #                               patience=10,
        #                               mode='min',
        #                               factor=self.lr_decay_factor,
        #                               threshold=3e-4)
        scheduler = OneCycleLR(optimizer=optimizer,
                               max_lr=self.lr,
                               total_steps=self.trainer.max_steps)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def log_images(self, batch, outputs, sample_steps=50):
        log = dict()

        input_low = batch["hint"]
        input_high = batch["jpg"]

        input_low, input_high, results, Astack = outputs
        alphas = torch.split(Astack, 3, 1)

        B, C, H, W = input_low.shape

        for b in range(min(2, B)):

            batch_list = [
                (input_low[b] + 1) / 2, # lq
                (input_high[b] + 1) / 2, # hq
            ]

            for i in range(len(self.return_result_index)):
                batch_list.append((alphas[self.return_result_index[i] - 1][b] + 1) / 2)
                batch_list.append((results[i][b] + 1) / 2)
            
            log[f"b{b}"] = torch.stack(batch_list, dim=0)

        return log

    def exposure_control_loss(self, enhances, rsize=16, E=0.6):
        E = (E - 0.5) / 0.5
        avg_intensity = F.avg_pool2d(enhances, rsize).mean(1)  # to gray: (R + G + B) / 3
        exp_loss = (avg_intensity - E).abs().mean()
        return exp_loss

    # Color constancy loss via gray-world assumption.   In use.
    def color_constency_loss(self, enhances):
        plane_avg = enhances.mean((2, 3))
        col_loss = torch.mean((plane_avg[:, 0] - plane_avg[:, 1]) ** 2
                            + (plane_avg[:, 1] - plane_avg[:, 2]) ** 2
                            + (plane_avg[:, 2] - plane_avg[:, 0]) ** 2)
        return col_loss

    # Averaged color component ratio preserving loss.  Not in use.
    def color_constency_loss2(self, enhances, originals):
        enh_cols = enhances.mean((2, 3))
        ori_cols = originals.mean((2, 3))
        rg_ratio = (enh_cols[:, 0] / enh_cols[:, 1] - ori_cols[:, 0] / ori_cols[:, 1]).abs()
        gb_ratio = (enh_cols[:, 1] / enh_cols[:, 2] - ori_cols[:, 1] / ori_cols[:, 2]).abs()
        br_ratio = (enh_cols[:, 2] / enh_cols[:, 0] - ori_cols[:, 2] / ori_cols[:, 0]).abs()
        col_loss = (rg_ratio + gb_ratio + br_ratio).mean()
        return col_loss
    
    def alpha_total_variation(self, A):
        '''
        Links: https://remi.flamary.com/demos/proxtv.html
            https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html#total_variation
        '''
        delta_h = A[:, :, 1:, :] - A[:, :, :-1, :]
        delta_w = A[:, :, :, 1:] - A[:, :, :, :-1]

        # TV used here: L-1 norm, sum R,G,B independently
        # Other variation of TV loss can be found by google search
        tv = delta_h.abs().mean((2, 3)) + delta_w.abs().mean((2, 3))
        loss = torch.mean(tv.sum(1) / (A.shape[1] / 3))
        return loss

    def spatial_consistency_loss(self, enhances, originals, to_gray, neigh_diff, rsize=4):
        # convert to gray
        enh_gray = F.conv2d(enhances, to_gray)
        ori_gray = F.conv2d(originals, to_gray)

        # average intensity of local regision
        enh_avg = F.avg_pool2d(enh_gray, rsize)
        ori_avg = F.avg_pool2d(ori_gray, rsize)

        # calculate spatial consistency loss via convolution
        enh_pad = F.pad(enh_avg, (1, 1, 1, 1), mode='replicate')
        ori_pad = F.pad(ori_avg, (1, 1, 1, 1), mode='replicate')
        enh_diff = F.conv2d(enh_pad, neigh_diff)
        ori_diff = F.conv2d(ori_pad, neigh_diff)

        spa_loss = torch.pow((enh_diff - ori_diff), 2).sum(1).mean()
        return spa_loss
    
    def get_kernels(self):
        # weighted RGB to gray
        K1 = torch.tensor([0.3, 0.59, 0.1], dtype=torch.float32).view(1, 3, 1, 1)

        # kernel for neighbor diff
        K2 = torch.tensor([[[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
                        [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 1, -1], [0, 0, 0]]], dtype=torch.float32)
        K2 = K2.unsqueeze(1)

        return K1, K2
