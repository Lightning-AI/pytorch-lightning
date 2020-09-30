"""
Runs either `.fit()` or `.test()` on a single node across multiple gpus.
"""
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from tests.base import EvalModelTemplate
import os
import torch


def main():
    seed_everything(1234)
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--trainer_method', default='fit')
    parser.add_argument('--tmpdir')
    parser.set_defaults(gpus=2)
    parser.set_defaults(distributed_backend="ddp")
    args = parser.parse_args()

    model = EvalModelTemplate()
    trainer = Trainer.from_argparse_args(args)

    result = {}
    if args.trainer_method == 'fit':
        print('-' * 100)
        print('FITTING')
        print('-' * 100)
        trainer.fit(model)
        print('-' * 100)
        print('DONE FITTING')
        print('-' * 100)
        result = {'status': 'complete', 'method': args.trainer_method, 'result': None}
    if args.trainer_method == 'test':
        result = trainer.test(model)
        result = {'status': 'complete', 'method': args.trainer_method, 'result': result}
    if args.trainer_method == 'fit_test':
        trainer.fit(model)
        result = trainer.test(model)
        result = {'status': 'complete', 'method': args.trainer_method, 'result': result}

    if len(result) > 0:
        file_path = os.path.join(args.tmpdir, 'ddp.result')
        torch.save(result, file_path)


if __name__ == '__main__':
    main()
