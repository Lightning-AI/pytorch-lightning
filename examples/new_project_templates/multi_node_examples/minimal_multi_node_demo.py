from pytorch_lightning import Trainer
from test_tube import Experiment
import os


def main():
    # use the cool model from the main README.md
    model = CoolModel()  # noqa: F821
    exp = Experiment(save_dir=os.getcwd())

    # train on 4 GPUs across 4 nodes
    trainer = Trainer(
        experiment=exp,
        distributed_backend='ddp',
        max_nb_epochs=10,
        gpus=4,
        nb_gpu_nodes=4
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()
