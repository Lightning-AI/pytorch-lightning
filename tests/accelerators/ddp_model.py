# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Runs either `.fit()` or `.test()` on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser

import torch

from pytorch_lightning import seed_everything, Trainer
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.simple_models import ClassificationModel


def main():
    seed_everything(4321)

    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--trainer_method', default='fit')
    parser.add_argument('--tmpdir')
    parser.add_argument('--workdir')
    parser.set_defaults(gpus=2)
    parser.set_defaults(accelerator="ddp")
    args = parser.parse_args()

    dm = ClassifDataModule()
    model = ClassificationModel()
    trainer = Trainer.from_argparse_args(args)

    if args.trainer_method == 'fit':
        trainer.fit(model, datamodule=dm)
        result = None
    elif args.trainer_method == 'test':
        result = trainer.test(model, datamodule=dm)
    elif args.trainer_method == 'fit_test':
        trainer.fit(model, datamodule=dm)
        result = trainer.test(model, datamodule=dm)
    else:
        raise ValueError(f'Unsupported: {args.trainer_method}')

    result_ext = {
        'status': 'complete',
        'method': args.trainer_method,
        'result': result,
    }
    file_path = os.path.join(args.tmpdir, 'ddp.result')
    torch.save(result_ext, file_path)


if __name__ == '__main__':
    main()
