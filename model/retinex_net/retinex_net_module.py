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
from .retinex_net import RetinexNet

class RetinexNetModule(BaseModelModule):

    def __init__(self, lr:float=0.001):
        super().__init__()
        self.lr = lr
        
        self.model = RetinexNet()

        self.loss = nn.L1Loss()

        self.metric_loe = LOE()
        self.metric_niqe = NIQE()
        self.metric_psnr = PSNR()
        self.metric_psnr_b = PSNRB()
        self.metric_ssim = SSIM()

        self.automatic_optimization = False

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
        
        R_low, I_low   = self.model.decom_net(input_low)
        R_high, I_high = self.model.decom_net(input_high)

        # Forward RelightNet
        I_delta = self.model.relight_net(I_low, R_low)

        # Other variables
        I_low_3  = torch.cat((I_low, I_low, I_low), dim=1)
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        I_delta_3= torch.cat((I_delta, I_delta, I_delta), dim=1)

        # Compute losses
        recon_loss_low = self.loss(R_low * I_low_3,  input_low)
        recon_loss_high = self.loss(R_high * I_high_3,  input_high)
        
        recon_loss_mutal_low = self.loss(R_high * I_low_3,  input_low)
        recon_loss_mutal_high = self.loss(R_low * I_high_3,  input_high)

        equal_r_loss = self.loss(R_low, R_high.detach())

        relight_loss = self.loss(R_low * I_delta_3, input_high)

        i_smooth_loss_low = self.smooth(I_low, R_low)
        i_smooth_loss_high = self.smooth(I_high, R_high)
        i_smooth_loss_delta = self.smooth(I_delta, R_low)

        loss_decom = \
            recon_loss_low + \
            recon_loss_high + \
            0.001 * recon_loss_mutal_low + \
            0.001 * recon_loss_mutal_high + \
            0.1 * i_smooth_loss_low + \
            0.1 * i_smooth_loss_high + \
            0.01 * equal_r_loss
        
        loss_relight = \
            relight_loss + \
            3 * i_smooth_loss_delta
        
        return loss_decom + loss_relight

    def training_step(self, batch, batch_idx):

        op_decom, op_relight = self.optimizers()
        sch_decom, sch_relight = self.lr_schedulers()

        # Train decom

        loss_decom = self(batch)

        op_decom.zero_grad()
        self.manual_backward(loss_decom)
        op_decom.step()
        sch_decom.step()

        # Train relight

        loss_relight = self(batch)
        
        self.log(name="train/loss", value=loss_relight)

        op_relight.zero_grad()
        self.manual_backward(loss_relight)
        op_relight.step()
        sch_relight.step()
    
    def validation_step(self, batch, batch_idx):

        input_low = batch["hint"]
        input_high = batch["jpg"]
        
        R_low, I_low = self.model.decom_net(input_low)
        I_low_3  = torch.cat((I_low, I_low, I_low), dim=1)

        I_delta = self.model.relight_net(I_low, R_low)
        I_delta_3= torch.cat((I_delta, I_delta, I_delta), dim=1)

        relight = R_low * I_delta_3

        metric_loe = self.metric_loe(relight, input_high).mean()
        metric_niqe = self.metric_niqe(relight).mean()
        metric_psnr = self.metric_psnr(relight, input_high).mean()
        metric_psnr_b = self.metric_psnr_b(relight, input_high).mean()
        metric_ssim = self.metric_ssim(relight, input_high).mean()

        self.log(name="val/metric_loe", value=metric_loe)
        self.log(name="val/metric_niqe", value=metric_niqe)
        self.log(name="val/metric_psnr", value=metric_psnr)
        self.log(name="val/metric_psnr_b", value=metric_psnr_b)
        self.log(name="val/metric_ssim", value=metric_ssim)

        return R_low, I_low_3, I_delta_3, relight
    
    def forward_enhance(self, batch, batch_idx):
        input_low = batch["hint"]
        
        R_low, I_low = self.model.decom_net(input_low)
        I_low_3  = torch.cat((I_low, I_low, I_low), dim=1)

        I_delta = self.model.relight_net(I_low, R_low)
        I_delta_3= torch.cat((I_delta, I_delta, I_delta), dim=1)

        input_enhance = R_low * I_delta_3

        return input_enhance

    def configure_optimizers(self):
        op_decom   = Adam(self.model.decom_net.parameters(),
                          lr=self.lr,
                          betas=(0.9, 0.999))
        op_relight = Adam(self.model.relight_net.parameters(),
                          lr=self.lr,
                          betas=(0.9, 0.999))
        sch_decom = OneCycleLR(optimizer=op_decom,
                               max_lr=self.lr,
                               total_steps=self.trainer.max_steps)
        sch_relight = OneCycleLR(optimizer=op_relight,
                                 max_lr=self.lr,
                                 total_steps=self.trainer.max_steps)

        return [op_decom, op_relight], [sch_decom, sch_relight]
    
    def log_images(self, batch, outputs, sample_steps=50):
        log = dict()

        input_low = batch["hint"]
        input_high = batch["jpg"]

        R_low, I_low, I_delta_3, relight = outputs

        B, C, H, W = input_low.shape

        for b in range(min(2, B)):
            
            log[f"b{b}"] = torch.stack(
                [
                    (input_low[b] + 1) / 2, # lq
                    (input_high[b] + 1) / 2, # hq
                    (R_low[b] + 1) / 2, # 
                    (I_low[b] + 1) / 2, # 
                    (I_delta_3[b] + 1) / 2, # relight
                    (relight[b] + 1) / 2, # restoration
                ],
                dim=0
            )

        return log

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).to(self.device)
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(input=self.gradient(input_tensor, direction),
                            kernel_size=3,
                            stride=1,
                            padding=1)

