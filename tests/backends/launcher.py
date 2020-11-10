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
import sys
import os
import coverage
import subprocess
from subprocess import TimeoutExpired
from pathlib import Path
import pytorch_lightning
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from tests.base import EvalModelTemplate
import torch
import functools
import itertools

def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def call_training_script(cli_args, tmpdir, env, timeout=20):
    file = Path(__file__).absolute()
    cli_args = cli_args.split(' ') if cli_args else []
    cli_args += ['--tmpdir', str(tmpdir)]
    command = [sys.executable, str(file)] + cli_args

    # need to set the PYTHONPATH in case pytorch_lightning was not installed into the environment
    env['PYTHONPATH'] = f'{pytorch_lightning.__file__}:' + env.get('PYTHONPATH', '')

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

def create_cmd_lines(cmd_line, **kwargs):
    keys = kwargs.keys()
    keys = sorted(keys)
    values_comb = itertools.product(*(kwargs[k] for k in keys))
    cmd_lines = []
    for combi in values_comb:
        temp_cmd_line = cmd_line[::]
        for key_idx, v in enumerate(combi):
            k = keys[key_idx]
            temp_cmd_line = temp_cmd_line.replace(f"[{k}]", str(v))
        cmd_lines.append(temp_cmd_line)
    cmd_lines = list(set(cmd_lines))
    return cmd_lines

class DDPLauncher:
    """ Class used to run ddp tests
    """
    @staticmethod
    def run_from_cmd_line(cli_args:str = None, func_to_run: callable = None, tmpdir: str = None, timeout=20):
        env = os.environ.copy()
        env["PL_CURRENT_TEST_MODULE"] = str(func_to_run.__module__)
        env["PL_CURRENT_TEST_NAME"] = str(func_to_run.__name__)
        call_training_script(cli_args, tmpdir, env, timeout=20)

    def run(cmd_line, **kwargs):
        cmd_lines = create_cmd_lines(cmd_line, **kwargs)
        def inner(func):
            @functools.wraps(func)
            def func_wrapper(*args, **kwargs):
                for cmd_line in cmd_lines:
                    print(f"Launching {func.__name__} with {cmd_line}")
                    DDPLauncher.run_from_cmd_line(cmd_line, func, os.getcwd())
            return func_wrapper
        return inner


if __name__ == "__main__":
    seed_everything(1234)
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--tmpdir')
    parser.set_defaults(gpus=2)
    args = parser.parse_args()
    os.environ["PL_IN_LAUNCHER"] = '1'
    env = os.environ.copy()
    func = import_from(env["PL_CURRENT_TEST_MODULE"], env["PL_CURRENT_TEST_NAME"])
    result = func(args)
    result = {'status': 'complete', 'result':result}
    if len(result) > 0:
        file_path = os.path.join(args.tmpdir, 'ddp.result')
        torch.save(result, file_path)
