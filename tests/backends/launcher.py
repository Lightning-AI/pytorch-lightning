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
    command = [sys.executable, '-m', 'coverage', 'run', str(file)] + cli_args

    # need to set the PYTHONPATH in case pytorch_lightning was not installed into the environment
    env['PYTHONPATH'] = f'{pytorch_lightning.__file__}:{env.get("PYTHONPATH", "")}'

    # for running in ddp mode, we need to lauch it's own process or pytest will get stuck
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
    """
    This script is used to launch DDP related tests.
    It provides a simple decorator to run your test. See below for explication and example:

    Example:

        # The decorator will read cmd_line + arguments provided as kwargs.

        @DDPLauncher.run("--max_epochs [max_epochs] --gpus 2 --accelerator [accelerator]",
                         max_epochs=["1"],
                         accelerator=["ddp", "ddp_spawn"])
        def test_cli_to_pass(tmpdir, args=None):

            ... do something with args + BoringModel

            return '1'


    Explication:

        1 - DDPLauncher.run will recieve a command line to run where tokens are recognized by [].
            DDPLauncher.run will emulate pytest `parametrize` function and generate as many
            resolved cmd_lines from your provided cmd_line + product of your kwargs arguments.
            For the previous example, it will generate 2 cmd_lines and run them:
                1: --max_epochs 1 --gpus 2 --accelerator ddp
                2: --max_epochs 1 --gpus 2 --accelerator ddp_spawn

        2 - For each cmd_line, the launcher will save the module and function name of your decorated test
            in env variable and run the cmd_line on himself, which is located at tests/backend/launcher.py.

        3 - When running {ENV} python tests/backend/launcher.py {your_resolved_cmd_line},
            the script will start in `__name__ == "__main__"` where argparse
            is used to parsed your command line.
            The parsed_args will be provided to `main` function.

        4 - the `main function` will extract module and function name of your decorated test
            from os.environ variables, import dymically your function, undecorate it,
            and run result = your_test_func(tmpdir, args=args).

        5 - the `main function` will save a `ddp.result` object which will be read by
            the launcher to make sure your test run correctly.
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
                    std, err = DDPLauncher.run_from_cmd_line(cmd_line, func, tmpdir, timeout=20)
                    print(std)
                    print(err)
                    # Make sure the test run properly
                    result_path = os.path.join(tmpdir, 'ddp.result')
                    result = torch.load(result_path)
                    # verify the file wrote the expected outputs
                    assert result['status'] == 'complete'
            return func_wrapper
        return inner


def main(args):
    # Set PL_IN_LAUNCHER for first use case
    os.environ["PL_IN_LAUNCHER"] = '1'
    env = os.environ.copy()

    # Load function based on module and its name
    func = import_from(env["PL_CURRENT_TEST_MODULE"], env["PL_CURRENT_TEST_NAME"])

    # Undecorate the function
    func = undecorated(func)

    # Run the function and gather result
    result = func(args.tmpdir, args=args)

    # Save result
    result = {'status': 'complete', 'result':result}
    if len(result) > 0:
        file_path = os.path.join(args.tmpdir, 'ddp.result')
        torch.save(result, file_path)


if __name__ == "__main__":
    seed_everything(1234)

    # Parse arguments
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--tmpdir')
    parser.set_defaults(gpus=2)

    # Launch main process
    main(parser.parse_args())
