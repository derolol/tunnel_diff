import einops
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning import LightningModule

from model.segformer.segformer import SegFormer
from model.segformer.focal_loss import FocalLoss
from metric.seg_accuracy import SegAccuracyMetric
from metric.seg_precision import SegPrecisionMetric
from metric.seg_iou import SegIoUMetric
from metric.seg_fbeta_score import SegFBetaScoreMetric
from metric.seg_recall import SegRecallMetric

def map_color(color_map, annotation):
    '''
    input: [BHW]
    '''
    annotation = color_map[annotation.long()]
    return einops.rearrange(annotation, 'b h w c -> b c h w')

class SegFormerModule(LightningModule):

    def __init__(self,
                 lr,
                 num_classes,
                 type_classes,
                 color_map,
                 pretrained):
        
        super().__init__()

        self.lr = lr
        self.num_classes = num_classes
        self.type_classes = type_classes
        self.color_map = color_map
        self.pretrained = pretrained
        
        self.model = SegFormer(num_classes=num_classes,
                               pretrained_path=pretrained,
                               in_channels=3,
                               embed_dims=[32, 64, 160, 256],
                               num_heads=[1, 2, 5, 8],
                               mlp_ratios=[4, 4, 4, 4],
                               qkv_bias=True,
                               depths=[2, 2, 2, 2],
                               sr_ratios=[8, 4, 2, 1],
                               drop_rate=0.0,
                               drop_path_rate=0.1,
                               head_embed_dim=256)
        
        self.loss = FocalLoss()

        self.metric_accuracy = SegAccuracyMetric(num_classes=self.num_classes)
        self.metric_f1score = SegFBetaScoreMetric(num_classes=self.num_classes)
        self.metric_precision = SegPrecisionMetric(num_classes=self.num_classes)
        self.metric_recall = SegRecallMetric(num_classes=self.num_classes)
        self.metric_iou = SegIoUMetric(num_classes=self.num_classes)
    
    def log(self, name, value):
        super().log(name=name,
                    value=value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    logger=True)

    def forward(self, batch):

        image = batch["hint"]
        annotation = batch["annotation"]

        output = self.model(image)
        loss = self.loss(output, annotation)

        return loss, output

    def training_step(self, batch, batch_idx):

        loss, output = self(batch)

        self.log("train/loss", loss)

        return {"loss": loss,
                "output": output}
    
    def validation_step(self, batch, batch_idx):
        
        loss, output = self(batch)
        annotation = batch["annotation"]

        metric_accuracy = self.metric_accuracy(output, annotation).float().mean(dim=0)
        metric_f1score = self.metric_f1score(output, annotation).float().mean(dim=0)
        metric_precision = self.metric_precision(output, annotation).float().mean(dim=0)
        metric_recall = self.metric_recall(output, annotation).float().mean(dim=0)
        metric_iou = self.metric_iou(output, annotation).float().mean(dim=0)

        for i in range(len(self.type_classes)):
            type_class = self.type_classes[i]
            self.log(f"val/metric_accuracy/{type_class}", metric_accuracy[i])
            self.log(f"val/metric_f1score/{type_class}", metric_f1score[i])
            self.log(f"val/metric_precision/{type_class}", metric_precision[i])
            self.log(f"val/metric_recall/{type_class}", metric_recall[i])
            self.log(f"val/metric_iou/{type_class}", metric_iou[i])
        
        self.log("val/metric_accuracy/total", metric_accuracy.mean())
        self.log("val/metric_f1score/total", metric_f1score.mean())
        self.log("val/metric_precision/total", metric_precision.mean())
        self.log("val/metric_recall/total", metric_recall.mean())
        self.log("val/metric_iou/total", metric_iou.mean())

        return {"loss": loss,
                "output": output}

    def configure_optimizers(self):
        
        optimizer = Adam(self.model.parameters(),
                         lr=self.lr)
        
        scheduler = OneCycleLR(optimizer=optimizer,
                               max_lr=self.lr,
                               total_steps=self.trainer.max_steps)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def log_images(self, batch, outputs, sample_steps=50):
        
        log = dict()

        image = batch["hint"]
        annotation = batch["annotation"]
        seg = outputs["output"].argmax(dim=1)

        B, C, H, W = image.shape

        color_map = torch.tensor(self.color_map).to(self.device)

        annotation_color = map_color(color_map, annotation).float() / 255
        seg_color = map_color(color_map, seg).float() / 255

        weight = 0.3
        weight_seg_color = image * (1 - weight) + seg_color * weight
        weight_seg_color = weight_seg_color.clamp(min=0, max=1.0)

        for b in range(min(2, B)):

            log[f"b{b}"] = torch.stack(
                [
                    (image[b] + 1) / 2, # image
                    annotation_color[b], # annoation
                    seg_color[b], # pred annoataion
                    weight_seg_color[b] # 
                ],
                dim=0)

        return log

    def test_on_tunnel_defect_enhance(self, batch, batch_idx):

        loss, output = self(batch)
        image = batch["hint"]
        annotation = batch["annotation"]
        image_name_list = batch["image_name"]

        B, C, H, W = image.shape

        metric_accuracy = self.metric_accuracy(output, annotation).float()
        metric_f1score = self.metric_f1score(output, annotation).float()
        metric_precision = self.metric_precision(output, annotation).float()
        metric_recall = self.metric_recall(output, annotation).float()
        metric_iou = self.metric_iou(output, annotation).float()

        rows = []
        for b in range(B):
            row = [
                image_name_list[b],
                "|".join([str(num) for num in metric_precision[b].tolist()]),
                "|".join([str(num) for num in metric_accuracy[b].tolist()]),
                "|".join([str(num) for num in metric_recall[b].tolist()]),
                "|".join([str(num) for num in metric_f1score[b].tolist()]),
                "|".join([str(num) for num in metric_iou[b].tolist()]),
            ]
            rows.append(row)

        seg = output.argmax(dim=1)
        color_map = torch.tensor(self.color_map).to(self.device)

        seg_color = map_color(color_map, seg).float() / 255

        weight_seg_color = image
        weight_seg_color[:, :, :][seg_color[:, :, :] > 0] = 1.0

        return rows, weight_seg_color