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
        err = str(err.decode("utf-8"))
        if 'Exception' in err:
            raise Exception(err)
    except TimeoutExpired:
        p.kill()
        std, err = p.communicate()

    return std, err

class DDPLauncher:
    """ Class used to run ddp tests
    """
    atol = 1e-8

    def run_from_cmd_line(cli_args:str = None, func_to_run: callable = None):
        if ddp:
            if sys.platform == "win32":
                pytest.skip("DDP not supported on windows")

        os.environ["PL_TEST_CODE"] = func_to_run.__code__

        call_training_script(__file__, cli_args)

if __name__ = "__main__":
    seed_everything(1234)
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--tmpdir')
    parser.set_defaults(gpus=2)
    args = parser.parse_args()
    code = os.get("PL_TEST_CODE", None)
    func = exec(code)
    result = func(args)
    result = {'status': 'complete', 'result': result}

    if len(result) > 0:
        file_path = os.path.join(args.tmpdir, 'ddp.result')
        torch.save(result, file_path)
