# Copyright The Lightning AI team.
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
import pytest
from lightning.fabric.plugins.io import TorchCheckpointIO


def test_remove_checkpoint(tmp_path):
    """Test that the IO can remove folders, files, and symlinks."""
    io = TorchCheckpointIO()

    # Path does not exist
    with pytest.raises(FileNotFoundError):
        io.remove_checkpoint("does_not_exist.txt")

    # Single file
    file = tmp_path / "file.txt"
    file.touch()
    io.remove_checkpoint(file)
    assert not file.exists()

    # Symlink
    file = tmp_path / "file.txt"
    file.touch()
    link = tmp_path / "link.txt"
    link.symlink_to(file)
    io.remove_checkpoint(link)
    assert file.exists()
    assert not link.is_symlink()
    file.unlink()

    # Broken Symlink
    file_not_exists = tmp_path / "not_exist.txt"
    link = tmp_path / "link.txt"
    link.symlink_to(file_not_exists)
    assert not file_not_exists.exists()
    io.remove_checkpoint(link)
    assert not link.is_symlink()

    # Folder with contents
    folder = tmp_path / "folder"
    nested_folder = folder / "nested_folder"
    nested_folder.mkdir(parents=True)
    file = nested_folder / "file.txt"
    file.touch()
    io.remove_checkpoint(folder)
    assert not folder.exists()
