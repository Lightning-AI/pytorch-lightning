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
from unittest import mock

import pytest

from pl_examples import _DALI_AVAILABLE
from tests.helpers.runif import RunIf

ARGS_DEFAULT = (
    "--trainer.default_root_dir %(tmpdir)s "
    "--trainer.max_epochs 1 "
    "--trainer.limit_train_batches 2 "
    "--trainer.limit_val_batches 2 "
    "--trainer.limit_test_batches 2 "
    "--trainer.limit_predict_batches 2 "
    "--data.batch_size 32 "
)
ARGS_GPU = ARGS_DEFAULT + "--trainer.gpus 1 "


@pytest.mark.skipif(not _DALI_AVAILABLE, reason="Nvidia DALI required")
@RunIf(min_gpus=1, skip_windows=True)
@pytest.mark.parametrize("cli_args", [ARGS_GPU])
def test_examples_mnist_dali(tmpdir, cli_args):
    from pl_examples.integration_examples.dali_image_classifier import cli_main

    # update the temp dir
    cli_args = cli_args % {"tmpdir": tmpdir}
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()
