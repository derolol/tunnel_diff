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

    model_module_class = get_obj_from_str(config.model["target"])
    model_module = model_module_class.load_from_checkpoint(config.model["checkpoint"], **config.model["params"])
    model_module.eval()

    # Create CSV
    log_save_dir = config.lightning.logger.params.save_dir
    log_name = config.lightning.logger.params.name
    log_path = os.path.join(log_save_dir, log_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    csv_name = f"test-{time.time()}"
    log_csv_path = os.path.join(log_path, f"{csv_name}.csv")
    os.mknod(log_csv_path)

    # Image path
    log_image_path = os.path.join(log_path, f"{csv_name}_image")
    if not os.path.exists(log_image_path):
        os.makedirs(log_image_path)

    with open(log_csv_path, "w") as csvfile: 

        writer = csv.writer(csvfile)
        writer.writerow(["image_name", "loe", "niqe", "psnr", "psnr_b", "ssim"])

        for batch_idx, batch in enumerate(tqdm(val_dataloader)):

            batch["jpg"] = batch["jpg"].to(model_module.device)
            batch["hint"] = batch["hint"].to(model_module.device)
            batch["low"] = batch["low"].to(model_module.device)
        
            with torch.no_grad():

                rows, images = model_module.test_on_tunnel_defect(batch, batch_idx)
            
            writer.writerows(rows) # [[item1, item2, item3], [row2], [row3]]
            for index in range(len(rows)):
                image_name = rows[index][0]
                output_tensor = images[index].clone().detach() # CHW
                output_tensor = (output_tensor + 1.0) / 2.0
                output_tensor = output_tensor.cpu()
                output_tensor = output_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
                output_tensor = cv2.cvtColor(output_tensor, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(log_image_path, f'{image_name}.png'), output_tensor)

if __name__ == "__main__":
    main()
