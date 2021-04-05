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
import os
import subprocess
import sys
from pathlib import Path
from subprocess import TimeoutExpired
from unittest import mock

import pytorch_lightning


def call_training_script(module_file, cli_args, method, tmpdir, timeout=60):
    file = Path(module_file.__file__).absolute()
    cli_args = cli_args.split(' ') if cli_args else []
    cli_args += ['--tmpdir', str(tmpdir)]
    cli_args += ['--trainer_method', method]
    command = [sys.executable, str(file)] + cli_args

    # need to set the PYTHONPATH in case pytorch_lightning was not installed into the environment
    env = os.environ.copy()
    env['PYTHONPATH'] = env.get('PYTHONPATH', '') + f'{pytorch_lightning.__file__}:'

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


@mock.patch.dict(os.environ, {"SLURM_PROCID": "0"})
def test_rank_zero_slurm():
    """ Test that SLURM environment variables are properly checked for rank_zero_only. """
    from pytorch_lightning.utilities.distributed import _get_rank, rank_zero_only
    rank_zero_only.rank = _get_rank()

    @rank_zero_only
    def foo():
        # The return type is optional because on non-zero ranks it will not be called
        return 1

    x = foo()
    assert x == 1


@mock.patch.dict(os.environ, {"RANK": "0"})
def test_rank_zero_torchelastic():
    """ Test that torchelastic environment variables are properly checked for rank_zero_only. """
    from pytorch_lightning.utilities.distributed import _get_rank, rank_zero_only
    rank_zero_only.rank = _get_rank()

    @rank_zero_only
    def foo():
        # The return type is optional because on non-zero ranks it will not be called
        return 1

    x = foo()
    assert x == 1


@mock.patch.dict(os.environ, {"RANK": "1", "SLURM_PROCID": "2", "LOCAL_RANK": "3"})
def test_rank_zero_none_set():
    """ Test that function is not called when rank environment variables are not global zero. """

    from pytorch_lightning.utilities.distributed import _get_rank, rank_zero_only
    rank_zero_only.rank = _get_rank()

    @rank_zero_only
    def foo():
        return 1

    x = foo()
    assert x is None
