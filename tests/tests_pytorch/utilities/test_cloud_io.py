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

import fsspec
from fsspec.implementations.local import LocalFileSystem

from pytorch_lightning.utilities.cloud_io import get_filesystem


def test_get_filesystem_custom_filesystem():
    _DUMMY_PRFEIX = "dummy"

    class DummyFileSystem(LocalFileSystem):
        ...

    fsspec.register_implementation(_DUMMY_PRFEIX, DummyFileSystem, clobber=True)
    output_file = os.path.join(f"{_DUMMY_PRFEIX}://", "tmpdir/tmp_file")
    assert isinstance(get_filesystem(output_file), DummyFileSystem)


def test_get_filesystem_local_filesystem():
    assert isinstance(get_filesystem("tmpdir/tmp_file"), LocalFileSystem)
