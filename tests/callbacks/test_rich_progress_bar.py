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

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ProgressBarBase, RichProgressBar
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


@RunIf(rich=True)
def test_rich_progress_bar_callback():

    trainer = Trainer(callbacks=RichProgressBar())

    progress_bars = [c for c in trainer.callbacks if isinstance(c, ProgressBarBase)]

    assert len(progress_bars) == 1
    assert isinstance(trainer.progress_bar_callback, RichProgressBar)


@RunIf(rich=True)
@mock.patch("pytorch_lightning.callbacks.progress.rich_progress.Progress.update")
def test_rich_progress_bar(progress_update, tmpdir):

    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        max_steps=1,
        callbacks=RichProgressBar(),
    )

    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model)

    assert progress_update.call_count == 6


def test_rich_progress_bar_import_error():

    with pytest.raises(ImportError, match="`RichProgressBar` requires `rich` to be installed."):
        Trainer(callbacks=RichProgressBar())
