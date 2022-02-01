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

from collections import defaultdict

import pytest

from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.trainer.trainer import Trainer
from tests.helpers.boring_model import BoringModel


@pytest.mark.parametrize("progress_base_class", [TQDMProgressBar, RichProgressBar])
def test_progress_bar_correct_value_epoch_end(tmpdir, progress_base_class):
    class MockedProgressBar(progress_base_class):
        calls = defaultdict(list)

        def get_metrics(self, trainer, pl_module):
            items = super().get_metrics(trainer, model)
            del items["v_num"]
            del items["loss"]
            # this is equivalent to mocking `set_postfix` as this method gets called every time
            self.calls[trainer.state.fn].append(
                (trainer.state.stage, trainer.current_epoch, trainer.global_step, items)
            )
            return items

    class MyModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("a", self.global_step, prog_bar=True, on_step=False, on_epoch=True, reduce_fx=max)
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            self.log("b", self.global_step, prog_bar=True, on_step=False, on_epoch=True, reduce_fx=max)
            return super().validation_step(batch, batch_idx)

        def test_step(self, batch, batch_idx):
            self.log("c", self.global_step, prog_bar=True, on_step=False, on_epoch=True, reduce_fx=max)
            return super().test_step(batch, batch_idx)

    model = MyModel()
    pbar = MockedProgressBar()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=2,
        enable_model_summary=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
        callbacks=pbar,
    )

    trainer.fit(model)
    assert pbar.calls["fit"] == [
        ("sanity_check", 0, 0, {"b": 0}),
        ("train", 0, 0, {}),
        ("train", 0, 1, {}),
        ("validate", 0, 1, {"b": 1}),  # validation end
        # epoch end over, `on_epoch=True` metrics are computed
        ("train", 0, 2, {"a": 1, "b": 1}),  # training epoch end
        ("train", 1, 2, {"a": 1, "b": 1}),
        ("train", 1, 3, {"a": 1, "b": 1}),
        ("validate", 1, 3, {"a": 1, "b": 3}),  # validation end
        ("train", 1, 4, {"a": 3, "b": 3}),  # training epoch end
    ]

    trainer.validate(model, verbose=False)
    assert pbar.calls["validate"] == []

    trainer.test(model, verbose=False)
    assert pbar.calls["test"] == []
