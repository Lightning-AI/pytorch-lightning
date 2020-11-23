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
import functools
import itertools
import os
import subprocess
import sys
from argparse import ArgumentParser
from inspect import isclass, isfunction, ismethod
from pathlib import Path
from subprocess import TimeoutExpired
from time import time
from typing import Callable, Dict, Optional

import coverage
import torch

import pytorch_lightning
from pytorch_lightning import Trainer, seed_everything
from tests.base import EvalModelTemplate


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def call_training_script(cli_args: str, tmpdir: str, env: Dict, timeout: int = 20):
    file = Path(__file__).absolute()
    cli_args = cli_args.split(' ') if cli_args else []
    cli_args += ['--tmpdir', str(tmpdir)]
    command = [sys.executable, '-m', 'coverage', 'run', '--source', 'pytorch_lightning', str(file)] + cli_args
    env['PYTHONPATH'] = f'{pytorch_lightning.__file__}:{env.get("PYTHONPATH", "")}'
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        std, err = p.communicate(timeout=timeout)
        err = str(err.decode("utf-8"))
        if 'Exception' in err or 'Error' in err:
            raise Exception(err)
    except TimeoutExpired:
        p.kill()
        std, err = p.communicate()
    return std, err


def create_runs(**kwargs):
    run = ''
    keys = sorted(kwargs.keys())
    for key in keys:
        run += f'--{key} [{key}]'
    values_comb = itertools.product(*(kwargs[k] for k in keys))
    runs = []
    for combi in values_comb:
        temp_run = run[::]
        for key_idx, v in enumerate(combi):
            k = keys[key_idx]
            temp_run = temp_run.replace(f"[{k}]", str(v))
        runs.append(temp_run)
    runs = list(set(runs))
    return runs


def prune(o):
    if type(o) is type:
        return o

    try:
        closure = o.__closure__
    except AttributeError:
        return

    if closure:
        for cell in closure:
            if cell.cell_contents is o:
                continue

            if is_decorator(cell.cell_contents):
                func = prune(cell.cell_contents)
                if func:
                    return func
        else:
            return o
    else:
        return o


def is_decorator(a):
    return (
        isfunction(a) or ismethod(a) or isclass(a)
    )


class Tester:
    @staticmethod
    def run_from_str(cli_args:str = None, func_to_run: Optional[Callable] = None, tmpdir: Optional[str] = None, timeout: int = 20):
        env = os.environ.copy()
        env["PL_CURRENT_TEST_MODULE"] = str(func_to_run.__module__)
        env["PL_CURRENT_TEST_NAME"] = str(func_to_run.__name__)
        return call_training_script(cli_args, tmpdir, env, timeout=timeout)

    def run(**kwargs):
        runs = create_runs(**kwargs)

        def inner(func):
            @functools.wraps(func)
            def func_wrapper(*args, **kwargs):
                tmpdir = kwargs.get("tmpdir")
                for run in runs:
                    print(f"Launching {func.__name__} with {run}")
                    std, err = Tester.run_from_str(run, func, tmpdir, timeout=20)
                    if std is not None:
                        try:
                            print(std.decode("utf-8"))
                        except Exception:
                            print(err)
                    if err is not None:
                        try:
                            print(err.decode("utf-8"))
                        except Exception:
                            print(err)
                    result_path = os.path.join(tmpdir, 'ddp.result')
                    result = torch.load(result_path)
                    assert result['status'] == 'complete'
            return func_wrapper
        return inner


def main(args):
    os.environ["PL_IN_Tester"] = '1'
    env = os.environ.copy()
    func = import_from(env["PL_CURRENT_TEST_MODULE"], env["PL_CURRENT_TEST_NAME"])
    func = prune(func)
    result = func(args.tmpdir, args=args)
    result = {'status': 'complete', 'result':result}
    if len(result) > 0:
        file_path = os.path.join(args.tmpdir, 'ddp.result')
        torch.save(result, file_path)


if __name__ == "__main__":
    seed_everything(1234)
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--tmpdir')
    parser.set_defaults(gpus=2)
    main(parser.parse_args())
