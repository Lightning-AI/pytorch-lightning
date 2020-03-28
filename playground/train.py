import argparse

import pytorch_lightning as pl

from model import CoolModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_filters', type=int, required=True)
    args = parser.parse_args()

    trainer = pl.Trainer(
        max_epochs=10,
        gpus='0',
        fast_dev_run=True
    )

    model = CoolModel({'n_filters': args.n_filters, 'a': {'b': 'c'}})
    trainer.fit(model)


if __name__ == "__main__":
    main()
