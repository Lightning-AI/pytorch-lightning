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
from time import time
from inspect import isfunction, ismethod, isclass


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def call_training_script(cli_args, tmpdir, env, timeout=20):
    file = Path(__file__).absolute()
    cli_args = cli_args.split(' ') if cli_args else []
    cli_args += ['--tmpdir', str(tmpdir)]
    command = [sys.executable, '-m', 'coverage', 'run', str(file)] + cli_args

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
    keys = sorted(kwargs.keys())
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


def undecorated(o):
    """Remove all decorators from a function, method or class"""
    # class decorator
    if type(o) is type:
        return o

    try:
        # python2
        closure = o.func_closure
    except AttributeError:
        pass

    try:
        # python3
        closure = o.__closure__
    except AttributeError:
        return

    if closure:
        for cell in closure:
            # avoid infinite recursion
            if cell.cell_contents is o:
                continue

            # check if the contents looks like a decorator; in that case
            # we need to go one level down into the dream, otherwise it
            # might just be a different closed-over variable, which we
            # can ignore.

            # Note: this favors supporting decorators defined without
            # @wraps to the detriment of function/method/class closures
            if looks_like_a_decorator(cell.cell_contents):
                undecd = undecorated(cell.cell_contents)
                if undecd:
                    return undecd
        else:
            return o
    else:
        return o


def looks_like_a_decorator(a):
    return (
        isfunction(a) or ismethod(a) or isclass(a)
    )


class DDPLauncher:
    """ Class used to run ddp tests
    """
    @staticmethod
    def run_from_cmd_line(cli_args:str = None, func_to_run: Optional[Callable] = None, tmpdir: Optional[str] = None, timeout: int = 20):
        env = os.environ.copy()
        env["PL_CURRENT_TEST_MODULE"] = str(func_to_run.__module__)
        env["PL_CURRENT_TEST_NAME"] = str(func_to_run.__name__)
        return call_training_script(cli_args, tmpdir, env, timeout=timeout)

    def run(cmd_line, **kwargs):
        cmd_lines = create_cmd_lines(cmd_line, **kwargs)

        def inner(func):
            @functools.wraps(func)
            def func_wrapper(*args, **kwargs):
                tmpdir = kwargs.get("tmpdir")
                for cmd_line in cmd_lines:
                    print(f"Launching {func.__name__} with {cmd_line}")
                    t0 = time()
                    std, err = DDPLauncher.run_from_cmd_line(cmd_line, func, tmpdir, timeout=20)
                    result_path = os.path.join(tmpdir, 'ddp.result')
                    result = torch.load(result_path)
                    # verify the file wrote the expected outputs
                    assert result['status'] == 'complete'
                    assert result['result'] == '1', result['result']
                    t1 = time()
                    print(t1 - t0)
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
    func = undecorated(func)
    result = func(args.tmpdir, args=args)
    result = {'status': 'complete', 'result':result}
    if len(result) > 0:
        file_path = os.path.join(args.tmpdir, 'ddp.result')
        torch.save(result, file_path)
