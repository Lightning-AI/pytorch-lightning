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

import pytorch_lightning as pl
from pytorch_lightning.utilities.migration import migrate_checkpoint


def test_migrate_checkpoint(monkeypatch):
    """Test that the correct migration function gets executed given the current version of the checkpoint."""
    # A checkpoint that is older than any migration point in the index
    old_checkpoint = {
        "pytorch-lightning_version": "0.0.0",
        "content": 123
    }
    new_checkpoint, call_order = _run_simple_migration(monkeypatch, old_checkpoint)
    assert call_order == ["one", "two", "three", "four"]
    assert new_checkpoint == {
        "pytorch-lightning_version": pl.__version__,
        "content": 123
    }

    # A checkpoint that is newer, but not the newest
    old_checkpoint = {
        "pytorch-lightning_version": "1.0.3",
        "content": 123
    }
    new_checkpoint, call_order = _run_simple_migration(monkeypatch, old_checkpoint)
    assert call_order == ["four"]
    assert new_checkpoint == {
        "pytorch-lightning_version": pl.__version__,
        "content": 123
    }

    # A checkpoint newer than any migration point in the index
    old_checkpoint = {
        "pytorch-lightning_version": "2.0",
        "content": 123
    }
    new_checkpoint, call_order = _run_simple_migration(monkeypatch, old_checkpoint)
    assert call_order == []
    assert new_checkpoint == {
        "pytorch-lightning_version": pl.__version__,
        "content": 123
    }


def _run_simple_migration(monkeypatch, old_checkpoint):
    call_order = []

    def dummy_upgrade(tag):
        def upgrade(ckpt):
            call_order.append(tag)
            return ckpt

        return upgrade

    index = {
        "0.0.1": [dummy_upgrade("one")],
        "0.0.2": [dummy_upgrade("two"), dummy_upgrade("three")],
        "1.2.3": [dummy_upgrade("four")],
    }
    monkeypatch.setattr(pl.utilities.migration.utils, "migration_index", lambda: index)
    new_checkpoint = migrate_checkpoint(old_checkpoint)
    return new_checkpoint, call_order
