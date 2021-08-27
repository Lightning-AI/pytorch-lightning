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
import importlib
import platform
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from subprocess import TimeoutExpired
from unittest import mock

import pytest

from pytorch_lightning.utilities.imports import _module_available
from tests.helpers.runif import RunIf

_DALI_AVAILABLE = _module_available("nvidia.dali")

ARGS_DEFAULT = (
    "--trainer.default_root_dir %(tmpdir)s "
    "--trainer.max_epochs 1 "
    "--trainer.limit_train_batches 2 "
    "--trainer.limit_val_batches 2 "
    "--data.batch_size 32 "
)
ARGS_GPU = ARGS_DEFAULT + "--trainer.gpus 1 "
ARGS_DP = ARGS_DEFAULT + "--trainer.gpus 2 --trainer.accelerator dp "
ARGS_AMP = "--trainer.precision 16 "


def run(tmpdir, import_cli, cli_args):
    file = Path(__file__).absolute()
    cli_args = cli_args % {"tmpdir": tmpdir}
    # this will execute this exact same file
    coverage = ["-m", "coverage", "run", "--source", "pytorch_lightning", "-a"]
    command = [sys.executable, *coverage, str(file), f"--import_cli={import_cli}", f"--cli_args={cli_args}"]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        std, err = p.communicate(timeout=60)
        print(std)
        err = str(err.decode("utf-8"))
        if "Exception" in err or "Error" in err:
            raise Exception(err)
    except TimeoutExpired:
        p.kill()
        std, err = p.communicate()
    return std, err


@RunIf(min_gpus=2)
@pytest.mark.parametrize("cli_args", [ARGS_DP, ARGS_DP + ARGS_AMP])
def test_examples_dp_simple_image_classifier(tmpdir, cli_args):
    run(tmpdir, "pl_examples.basic_examples.simple_image_classifier", cli_args)


@RunIf(min_gpus=2)
@pytest.mark.parametrize("cli_args", [ARGS_DP, ARGS_DP + ARGS_AMP])
def test_examples_dp_backbone_image_classifier(tmpdir, cli_args):
    run(tmpdir, "pl_examples.basic_examples.backbone_image_classifier", cli_args)


@RunIf(min_gpus=2)
@pytest.mark.parametrize("cli_args", [ARGS_DP, ARGS_DP + ARGS_AMP])
def test_examples_dp_autoencoder(tmpdir, cli_args):
    run(tmpdir, "pl_examples.basic_examples.autoencoder", cli_args)


@pytest.mark.parametrize("cli_args", [ARGS_DEFAULT])
def test_examples_cpu_simple_image_classifier(tmpdir, cli_args):
    run(tmpdir, "pl_examples.basic_examples.simple_image_classifier", cli_args)


@pytest.mark.parametrize("cli_args", [ARGS_DEFAULT])
def test_examples_cpu_backbone_image_classifier(tmpdir, cli_args):
    run(tmpdir, "pl_examples.basic_examples.backbone_image_classifier", cli_args)


@pytest.mark.parametrize("cli_args", [ARGS_DEFAULT])
def test_examples_cpu_autoencoder(tmpdir, cli_args):
    run(tmpdir, "pl_examples.basic_examples.autoencoder", cli_args)


@pytest.mark.skipif(not _DALI_AVAILABLE, reason="Nvidia DALI required")
@RunIf(min_gpus=1)
@pytest.mark.skipif(platform.system() != "Linux", reason="Only applies to Linux platform.")
@pytest.mark.parametrize("cli_args", [ARGS_GPU])
def test_examples_mnist_dali(tmpdir, cli_args):
    run(tmpdir, "pl_examples.basic_examples.dali_image_classifier", cli_args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--import_cli", type=str)
    parser.add_argument("--cli_args", type=str)
    args = parser.parse_args()
    module = importlib.import_module(args.import_cli)
    with mock.patch("argparse._sys.argv", ["any.py"] + args.cli_args.strip().split()):
        module.cli_main()
