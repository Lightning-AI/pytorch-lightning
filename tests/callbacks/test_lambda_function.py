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
import inspect

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import Callback, LambdaCallback
from tests.helpers.boring_model import BoringModel


def test_lambda_call(tmpdir):
    seed_everything(42)

    class CustomModel(BoringModel):

        def on_train_epoch_start(self):
            if self.current_epoch > 1:
                raise KeyboardInterrupt

    checker = set()
    hooks = [m for m, _ in inspect.getmembers(Callback, predicate=inspect.isfunction)]
    hooks_args = {h: (lambda x: lambda *args: checker.add(x))(h) for h in hooks}
    hooks_args["on_save_checkpoint"] = (lambda x: lambda *args: [checker.add(x)])("on_save_checkpoint")

    model = CustomModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[LambdaCallback(**hooks_args)],
    )
    results = trainer.fit(model)
    assert results

    model = CustomModel()
    ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        resume_from_checkpoint=ckpt_path,
        callbacks=[LambdaCallback(**hooks_args)],
    )
    results = trainer.fit(model)
    trainer.test(model)

    assert results
    assert checker == set(hooks)
