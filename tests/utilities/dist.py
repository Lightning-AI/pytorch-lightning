import os
import subprocess
from subprocess import TimeoutExpired
import sys
from pathlib import Path

import pytorch_lightning


def call_training_script(module_file, cli_args, method, tmpdir, timeout=60):
    file = Path(module_file.__file__).absolute()
    cli_args = cli_args.split(' ') if cli_args else []
    cli_args += ['--tmpdir', str(tmpdir)]
    cli_args += ['--trainer_method', method]
    command = [sys.executable, str(file)] + cli_args

    # need to set the PYTHONPATH in case pytorch_lightning was not installed into the environment
    env = os.environ.copy()
    env['PYTHONPATH'] = f'{pytorch_lightning.__file__}:' + env.get('PYTHONPATH', '')

    # for running in ddp mode, we need to lauch it's own process or pytest will get stuck
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    try:
        std, err = p.communicate(timeout=timeout)
        # bubble up the error to tests
        if len(err) > 0:
            err = str(err.decode("utf-8"))
            raise Exception(err)
    except TimeoutExpired:
        p.kill()

    return std, err
