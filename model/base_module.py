from pytorch_lightning import LightningModule

class BaseModelModule(LightningModule):

    def __init__(self):
        super().__init__()

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
    
    def log_images(self, batch, outputs, sample_steps=50):
        pass

    def forward_enhance(self, batch, batch_idx):
        pass

    def test_on_exposure_errors(self, batch, batch_idx):

        input_low = batch["hint"]
        input_high = batch["jpg"]
        image_name_list = batch["image_name"]

        B, C, H, W = input_low.shape
        
        input_enhance = self.forward_enhance(batch, batch_idx) 

        metric_loe = self.metric_loe(input_enhance, input_high)
        metric_niqe = self.metric_niqe(input_enhance)
        metric_psnr = self.metric_psnr(input_enhance, input_high)
        metric_psnr_b = self.metric_psnr_b(input_enhance, input_high)
        metric_ssim = self.metric_ssim(input_enhance, input_high)

        rows = []
        for b in range(B):
            row = [
                image_name_list[b],
                metric_loe[b].item(),
                metric_niqe[b].item(),
                metric_psnr[b].item(),
                metric_psnr_b[b].item(),
                metric_ssim[b].item(),
            ]
            rows.append(row)

        return rows, input_enhance
    

    def test_on_tunnel_defect(self, batch, batch_idx):

        input_low = batch["hint"]
        # input_high = batch["jpg"]
        image_name_list = batch["image_name"]

        B, C, H, W = input_low.shape
        
        input_enhance = self.forward_enhance(batch, batch_idx)

        # RGB -> GRAY
        input_enhance = \
            input_enhance[:, 0 : 1, :, :] * 0.299 + \
            input_enhance[:, 1 : 2, :, :] * 0.587 + \
            input_enhance[:, 2 : 3, :, :] * 0.114
        input_enhance = input_enhance.repeat(1, 3, 1, 1)
        input_enhance = input_enhance.clip(- 1, 1)

        metric_loe = self.metric_loe(input_low, input_enhance)
        metric_niqe = self.metric_niqe(input_enhance)
        metric_psnr = self.metric_psnr(input_low, input_enhance)
        metric_psnr_b = self.metric_psnr_b(input_low, input_enhance)
        metric_ssim = self.metric_ssim(input_low, input_enhance)

        rows = []
        for b in range(B):
            row = [
                image_name_list[b],
                metric_loe[b].item(),
                metric_niqe[b].item(),
                metric_psnr[b].item(),
                metric_psnr_b[b].item(),
                metric_ssim[b].item(),
            ]
            rows.append(row)

        return rows, input_enhance
    
