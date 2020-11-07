"""
Pytorch Lightning training using Hydra for configuration
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer

# Hydra defaults from non-yaml config stores (trainer, optimizer, scheduler)
import pl_examples.hydra_examples.conf_simple.trainer
#import pl_examples.hydra_examples.conf.optimizer
#import pl_examples.hydra_examples.conf.scheduler

# Original lightning template
from pl_examples.models.lightning_template import LightningTemplateModel


@hydra.main(config_path="conf_simple", config_name="config_simple")
def main(cfg: DictConfig):
    """
    Main training routine specific for this project
    :param cfg:
    """

    print(cfg.pretty())
    seed_everything(cfg.model.seed)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(**cfg.model,**cfg.misc)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer(**cfg.trainer)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == "__main__":
    main()
