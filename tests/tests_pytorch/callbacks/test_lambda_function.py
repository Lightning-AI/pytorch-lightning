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
from functools import partial

import pytest
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, LambdaCallback
from lightning.pytorch.demos.boring_classes import BoringModel

from tests_pytorch.models.test_hooks import get_members


def test_lambda_call(tmp_path):
    seed_everything(42)

    class CustomException(Exception):
        pass

    class CustomModel(BoringModel):
        def on_train_epoch_start(self):
            if self.current_epoch > 1:
                raise CustomException("Custom exception to trigger `on_exception` hooks")

    checker = set()

    def call(hook, *_, **__):
        checker.add(hook)

    hooks = get_members(Callback) - {"state_dict", "load_state_dict"}
    hooks_args = {h: partial(call, h) for h in hooks}

    model = CustomModel()

    # successful run
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[LambdaCallback(**hooks_args)],
    )
    trainer.fit(model)
    ckpt_path = trainer.checkpoint_callback.best_model_path

    # raises KeyboardInterrupt and loads from checkpoint
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=3,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        callbacks=[LambdaCallback(**hooks_args)],
    )
    with pytest.raises(CustomException):
        trainer.fit(model, ckpt_path=ckpt_path)
    trainer.test(model)
    trainer.predict(model)

    assert checker == hooks
