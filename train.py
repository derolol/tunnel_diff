from argparse import ArgumentParser
from omegaconf import OmegaConf

import torch
from pytorch_lightning import seed_everything, Trainer

from util.common import instantiate_from_config, load_state_dict

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)

    seed_everything(config.lightning.seed, workers=True)
    
    data_module = instantiate_from_config(config.data)
    model_module = instantiate_from_config(config.model)
    if config.model.get("resume"):
        load_state_dict(model_module, torch.load(config.model.resume, map_location="cpu"), strict=True)
    
    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    
    # add logger
    logger = instantiate_from_config(config.lightning.logger)

    trainer = Trainer(logger=logger, callbacks=callbacks, **config.lightning.trainer)
    trainer.fit(model_module, datamodule=data_module)

if __name__ == "__main__":
    main()
