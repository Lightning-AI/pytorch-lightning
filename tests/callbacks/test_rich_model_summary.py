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
import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from pytorch_lightning.utilities.imports import _RICH_AVAILABLE
from tests.helpers.runif import RunIf


@RunIf(rich=True)
def test_rich_model_summary_callback():

    trainer = Trainer(callbacks=RichProgressBar())

    assert any(isinstance(cb, RichModelSummary) for cb in trainer.callbacks)
    assert isinstance(trainer.progress_bar_callback, RichProgressBar)


def test_rich_progress_bar_import_error():

    if not _RICH_AVAILABLE:
        with pytest.raises(ImportError, match="`RichModelSummary` requires `rich` to be installed."):
            Trainer(callbacks=RichModelSummary())
