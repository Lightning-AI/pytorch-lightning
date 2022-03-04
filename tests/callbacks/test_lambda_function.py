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
from functools import partial

import pytest

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import Callback, LambdaCallback
from tests.helpers.boring_model import BoringModel
from tests.models.test_hooks import get_members


def test_lambda_call(tmpdir):
    seed_everything(42)

    class CustomModel(BoringModel):
        def on_train_epoch_start(self):
            if self.current_epoch > 1:
                raise KeyboardInterrupt

    checker = set()

    def call(hook, *_, **__):
        checker.add(hook)

    hooks = get_members(Callback) - {"state_dict", "load_state_dict"}
    hooks_args = {h: partial(call, h) for h in hooks}
    hooks_args["on_save_checkpoint"] = lambda *_: [checker.add("on_save_checkpoint")]

    model = CustomModel()

    # successful run
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[LambdaCallback(**hooks_args)],
    )
    with pytest.deprecated_call(match="on_keyboard_interrupt` callback hook was deprecated in v1.5"):
        trainer.fit(model)

    ckpt_path = trainer.checkpoint_callback.best_model_path

    # raises KeyboardInterrupt and loads from checkpoint
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        callbacks=[LambdaCallback(**hooks_args)],
    )
    with pytest.deprecated_call(match="on_keyboard_interrupt` callback hook was deprecated in v1.5"):
        trainer.fit(model, ckpt_path=ckpt_path)
    with pytest.deprecated_call(match="on_keyboard_interrupt` callback hook was deprecated in v1.5"):
        trainer.test(model)
    with pytest.deprecated_call(match="on_keyboard_interrupt` callback hook was deprecated in v1.5"):
        trainer.predict(model)

    assert checker == hooks
