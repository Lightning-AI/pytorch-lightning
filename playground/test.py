import argparse

import pytorch_lightning as pl

from model import CoolModel


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--hparams', type=str, default=None)
args = parser.parse_args()

model = CoolModel.load_from_checkpoint(
    checkpoint_path=args.ckpt,
    hparams_file=args.hparams
)
