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
from argparse import ArgumentParser
import coverage
from pytorch_lightning import Trainer, seed_everything
from tests.base import EvalModelTemplate
import os
import torch
from tests.backends.ddp import *

def test_runner(module_file, cli_args, method, tmpdir, timeout=60):
    file = Path(module_file.__file__).absolute()
    cli_args = cli_args.split(' ') if cli_args else []
    cli_args += ['--tmpdir', str(tmpdir)]
    cli_args += ['--trainer_method', method]
    command = [sys.executable, str(file)] + cli_args

    # need to set the PYTHONPATH in case pytorch_lightning was not installed into the environment
    env = os.environ.copy()
    env['PYTHONPATH'] = f'{pytorch_lightning.__file__}:' + env.get('PYTHONPATH', '')

    # use to track coverage
    coverage.process_startup()
    # for running in ddp mode, we need to lauch it's own process or pytest will get stuck
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    try:
        std, err = p.communicate(timeout=timeout)
        err = str(err.decode("utf-8"))
        if 'Exception' in err:
            raise Exception(err)
    except TimeoutExpired:
        p.kill()
        std, err = p.communicate()

    return std, err


if __name__ == '__main__':
    seed_everything(1234)
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--trainer_method', default='fit')
    parser.add_argument('--tmpdir')
    parser.set_defaults(gpus=2)
    parser.set_defaults(distributed_backend="ddp")
    args = parser.parse_args()

    func_name = args.func_name
    func = getattr(locals(), func_name, None)
    result = func(args)
    if len(result) > 0:
        file_path = os.path.join(args.tmpdir, 'ddp.result')
        torch.save(result, file_path)
