import math
from typing import Mapping, Any
import copy
from collections import OrderedDict
import einops
import torch
import torch.nn as nn
from torch.nn import functional as F

from model.tunnel_diff.ldm.models.diffusion.ddpm import LatentDiffusion
from model.tunnel_diff.ldm.util import default, instantiate_from_config
from model.tunnel_diff.ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from model.tunnel_diff.utils.common import frozen_module
from model.tunnel_diff.spaced_sampler import SpacedSampler
from model.tunnel_diff.tunnel_diff import ControlNet

from metric.loe import LOE
from metric.niqe.niqe import NIQE
from metric.psnr import PSNR, PSNRB
from metric.ssim import SSIM

def color_loss(output, gt,mask=None):
    img_ref = F.normalize(output, p = 2, dim = 1)
    ref_p = F.normalize(gt, p = 2, dim = 1)
    if mask!=None:
        img_ref=mask*img_ref
        ref_p*=mask
    loss_cos = 1 - torch.mean(F.cosine_similarity(img_ref, ref_p, dim=1))
    # loss_cos = self.mse(img_ref, ref_p)
    return loss_cos

def light_loss(output,gt,mask=None):
    #output = torch.mean(output, 1, keepdim=True)
    #gt=torch.mean(gt,1,keepdim=True)
    output =output[:, 0:1, :, :] * 0.299 + output[:, 1:2, :, :] * 0.587 + output[:, 2:3, :, :] * 0.114
    gt = gt[:, 0:1, :, :] * 0.299 + gt[:, 1:2, :, :] * 0.587 + gt[:, 2:3, :, :] * 0.114
    if mask != None:
        output*=mask
        gt*=mask
    loss=F.l1_loss(output,gt)
    return loss

class TunnelDiffModule(LatentDiffusion):

    def __init__(
        self,
        control_stage_config: Mapping[str, Any],
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        # preprocess_config,
        sample_light: int = 500,
        *args,
        **kwargs
    ) -> "TunnelDiffModule":
        
        super().__init__(*args, **kwargs)
        
        # instantiate control module
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        
        # instantiate preprocess module (SwinIR)
        # self.preprocess_model = instantiate_from_config(preprocess_config)
        # frozen_module(self.preprocess_model)
        
        # instantiate condition encoder, since our condition encoder has the same 
        # structure with AE encoder, we just make a copy of AE encoder. please
        # note that AE encoder's parameters has not been initialized here.
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)), # cond_encoder.encoder
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv)) # cond_encoder.quant_conv
        ]))

        # 计算位置编码并将其存储在pe张量中
        light_length = 1000
        dim = 128
        pe = torch.zeros(light_length, dim) # 创建一个max_len x d_model的全零张量
        position = torch.arange(0, light_length).unsqueeze(1) # 生成0到max_len-1的整数序列，并添加一个维度
        # 计算div_term，用于缩放不同位置的正弦和余弦函数
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        # 使用正弦和余弦函数生成位置编码，对于d_model的偶数索引，使用正弦函数；对于奇数索引，使用余弦函数。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.light_encode = pe.float()

        self.sample_light = sample_light

        # CLDM
        # rand_mat = np.random.randn(128, 128)
        # rand_otho_mat, _ = np.linalg.qr(rand_mat)
        # np.savetxt('rand_otho.txt', rand_otho_mat)
        # self.light_encode = torch.from_numpy(rand_otho_mat).float()
        # self.register_buffer('light_encode', light_encode)

        self.metric_loe = LOE()
        self.metric_niqe = NIQE()
        self.metric_psnr = PSNR()
        self.metric_psnr_b = PSNRB()
        self.metric_ssim = SSIM()

        frozen_module(self.cond_encoder)

    def apply_condition_encoder(self, control):
        # c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent_meanvar = self.cond_encoder(control)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor
        return c_latent
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # self.batch_value = batch

        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        # control = control.to(self.device)
        # control = einops.rearrange(control, 'b h w c -> b c h w')
        # control = control.to(memory_format=torch.contiguous_format).float()
        lq = control
        # apply preprocess model
        # control = self.preprocess_model(control)
        # apply condition encoder
        c_latent = self.apply_condition_encoder(control)

        low_light = batch["low"]
        # low_light = low_light.to(self.device)
        # low_light = einops.rearrange(low_light, 'b h w c -> b c h w')
        # low_light = low_light.to(memory_format=torch.contiguous_format).float()
        low_light_latent = self.apply_condition_encoder(low_light)

        light_high = ((batch["jpg"] + 1.) / 2.).mean([1, 2, 3])  # b * 1
        light_high = light_high * 1000.
        light_context = self.getEmbedding(light_high)

        # print(light_high, light_context.mean())

        return x, dict(c_crossattn=[light_context], c_latent=[c_latent, low_light_latent], lq=[lq], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_latent'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            # print("apply_model", cond['c_latent'][0].shape)
            # print("apply_model", cond['c_latent'][1].shape)
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_latent'], dim=1),
                timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    def getEmbedding(self, coord):
        embed = self.light_encode.to(coord)[coord.long()] # [b, 128]
        return embed.unsqueeze(dim=1)
    
    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)

        # 重建
        z_start_pred = self.predict_start_from_noise(x_noisy, t=t, noise=model_output)
        x_start_pred = self.decode_first_stage(z_start_pred)

        # gt = self.batch_value["jpg"]
        gt = self.decode_first_stage(x_start)

        col_loss = color_loss(x_start_pred, gt)
        loss_dict.update({f'{prefix}/col_loss': col_loss})
        # col_loss_weight = 100 if self.current_epoch >= 20 else 0
        col_loss_weight = 1.0
        loss += col_loss * col_loss_weight

        exposure_loss = light_loss(x_start_pred, gt)
        loss_dict.update({f'{prefix}/exposure_loss': exposure_loss})
        # exposure_loss_weight = 20 if self.current_epoch >= 20 else 0
        exposure_loss_weight = 1.0
        loss += exposure_loss * exposure_loss_weight

        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()

        z, c = self.get_input(batch, self.first_stage_key)

        c_lq = c["lq"][0]
        c_latent, s_latent = c["c_latent"]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        samples = self.sample_log(
            cond_img=[c_latent, s_latent],
            steps=sample_steps)
        
        for b in range(min(4, c_lq.shape[0])):
            
            log[f"b{b}"] = torch.stack(
                [
                    (self.decode_first_stage(z[b:b+1])[0] + 1) / 2, # hq
                    c_cat[b], # control / lq
                    (self.decode_first_stage(c_latent[b:b+1])[0] + 1) / 2, # decoded_control
                    (self.decode_first_stage(s_latent[b:b+1])[0] + 1) / 2, # decoded_snr_control
                    samples[b] # sample
                ],
                dim=0
            )

        return log
    
    @torch.no_grad()
    def sample_log(self, cond_img, steps):
        # sampler = SpacedSampler(self)
        # b, c, h, w = cond["c_concat"][0].shape
        # shape = (b, self.channels, h // 8, w // 8)
        # samples = sampler.sample(
        #     steps, shape, cond["c_concat"][0],
        #     positive_prompt="",
        #     negative_prompt="",
        #     # unconditional_guidance_scale=1.0,
        #     # unconditional_conditioning=None
        # )
        sampler = SpacedSampler(self)
        b, c, h, w = cond_img[0].shape
        shape = (b, self.channels, h, w)
        samples = sampler.sample(
            steps, shape,
            cond_img,
            positive_prompt=self.getEmbedding(torch.ones(size=(b,)).to(self.device) * 1000 / 2), 
            negative_prompt=self.getEmbedding(torch.ones(size=(b,)).to(self.device) * 1000 / 2),
            cfg_scale=1.0,
            color_fix_type="none"
            # color_fix_type="wavelet"
        )
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def validation_step(self, batch, batch_idx):
        # TODO: 
        pass

    def forward_enhance(self, batch, batch_idx):
        z, c = self.get_input(batch, self.first_stage_key)
        c_latent, s_latent = c["c_latent"]

        sampler = SpacedSampler(self)
        b, c, h, w = c_latent.shape
        samples = sampler.sample(
            steps=50,
            shape=(b, self.channels, h, w),
            cond_img=[c_latent, s_latent],
            positive_prompt=self.getEmbedding(torch.ones(size=(b,)).to(self.device) * self.sample_light), 
            negative_prompt=self.getEmbedding(torch.ones(size=(b,)).to(self.device) * self.sample_light),
            cfg_scale=1.0,
            color_fix_type="none"
        )
        
        return samples * 2.0 - 1.0