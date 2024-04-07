from argparse import ArgumentParser
from omegaconf import OmegaConf

import os
import csv
import time
import cv2
from tqdm import tqdm

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import CSVLogger

from util.common import instantiate_from_config, get_obj_from_str

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)

    seed_everything(config.lightning.seed, workers=True)
    
    data_module = instantiate_from_config(config.data)
    data_module.setup("fit")
    val_dataloader = data_module.val_dataloader()

    log_save_dir = config.lightning.logger.params.save_dir
    log_path = os.path.join(log_save_dir, "test_tunnel_defect_image")
    # log_hq_path = os.path.join(log_path, "hq")
    log_lq_path = os.path.join(log_path, "lq")
    log_lf_path = os.path.join(log_path, "lf")
    if not os.path.exists(log_path):
        # os.makedirs(log_hq_path)
        os.makedirs(log_lq_path)
        os.makedirs(log_lf_path)

    for batch_idx, batch in enumerate(tqdm(val_dataloader)):

        hq = batch["jpg"]
        lq = batch["hint"]
        lf = batch["low"]
        image_name_list = batch["image_name"]

        B, C, H, W = hq.shape

        for index in range(B):
            # save_tensor(hq[index], os.path.join(log_hq_path, f"{image_name_list[index]}.png"))
            save_tensor(lq[index], os.path.join(log_lq_path, f"{image_name_list[index]}.png"))
            save_tensor(lf[index], os.path.join(log_lf_path, f"{image_name_list[index]}.png"))
            
def save_tensor(input_tensor, save_path):
    output_tensor = input_tensor.clone().detach() # CHW
    output_tensor = (output_tensor + 1.0) / 2.0
    output_tensor = output_tensor.cpu()
    output_tensor = output_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    output_tensor = cv2.cvtColor(output_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, output_tensor)

if __name__ == "__main__":
    main()
