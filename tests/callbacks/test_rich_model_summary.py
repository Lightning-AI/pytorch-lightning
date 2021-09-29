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


@RunIf(rich=True)
@mock.patch("pytorch_lightning.callbacks.rich_model_summary.Console.print", autospec=True)
@mock.patch("pytorch_lightning.callbacks.rich_model_summary.Table.add_row", autospec=True)
def test_rich_summary_tuples(mock_table_add_row, mock_console):
    """Ensure that tuples are converted into string, and print is called correctly."""
    model_summary = RichModelSummary()

    summary_data = [("x", [0]), ("Name", ["layer"]), ("Type", ["type"]), ("Params", ["params"]), ("In sizes", [(1, 1)])]
    model_summary.summarize(summary_data=summary_data, total_parameters=1, trainable_parameters=1, model_size=1)
    assert mock_console.call_count == 2
    # assert that the input summary data was converted correctly
    args, kwargs = mock_table_add_row.call_args_list[0]
    assert args[1:] == ("0", "layer", "type", "params", "(1, 1)")
