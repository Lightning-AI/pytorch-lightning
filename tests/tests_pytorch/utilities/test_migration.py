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

import pytorch_lightning
from pytorch_lightning.utilities.migration import pl_legacy_patch


def test_patch_legacy_argparse_utils():
    with pl_legacy_patch():
        from pytorch_lightning.utilities import argparse_utils

        assert callable(argparse_utils._gpus_arg_default)
        assert "pytorch_lightning.utilities.argparse_utils" in sys.modules

    assert "pytorch_lightning.utilities.argparse_utils" not in sys.modules


def test_patch_legacy_gpus_arg_default():
    with pl_legacy_patch():
        from pytorch_lightning.utilities.argparse import _gpus_arg_default

        assert callable(_gpus_arg_default)
    assert not hasattr(pytorch_lightning.utilities.argparse, "_gpus_arg_default")
    assert not hasattr(pytorch_lightning.utilities.argparse, "_gpus_arg_default")
