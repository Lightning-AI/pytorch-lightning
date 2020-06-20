"""
Pytorch Lightning training using Hydra for configuration
"""

import hydra
import pl_examples.hydra_examples.user_config
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


from pl_examples.models.hydra_config_model import LightningTemplateModel
from pytorch_lightning import Callback, seed_everything, Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main training routine specific for this project
    :param cfg:
    """
    seed_everything(cfg.model.seed)
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(cfg.model, cfg.data, cfg.scheduler, cfg.opt)
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
 
    callbacks = [instantiate(c) for c in cfg.callbacks.callbacks_list] if  cfg.callbacks else []

    trainer = Trainer(
        **cfg.trainer,
        logger=instantiate(cfg.logger),
        profiler=instantiate(cfg.profiler),
        checkpoint_callback=instantiate(cfg.checkpoint),
        early_stop_callback=instantiate(cfg.early_stopping),
        callbacks=callbacks,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == "__main__":
    main()
