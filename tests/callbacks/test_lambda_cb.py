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
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LambdaCallback
from tests.base.boring_model import BoringModel


def test_lambda_call(tmpdir):
    seed_everything(42)

    checker = set()

    hooks = [
        "setup",
        "teardown",
        "on_init_start",
        "on_init_end",
        "on_fit_start",
        "on_fit_end",
        "on_train_batch_start",
        "on_train_batch_end",
        "on_train_epoch_start",
        "on_train_epoch_end",
        "on_validation_epoch_start",
        "on_validation_epoch_end",
        "on_test_epoch_start",
        "on_test_epoch_end",
        "on_epoch_start",
        "on_epoch_end",
        "on_batch_start",
        "on_batch_end",
        "on_validation_batch_start",
        "on_validation_batch_end",
        "on_test_batch_start",
        "on_test_batch_end",
        "on_train_start",
        "on_train_end",
        "on_test_start",
        "on_test_end",
    ]
    model = BoringModel()

    hooks_args = {h: (lambda x: lambda *args: checker.add(x))(h) for h in hooks}
    test_callback = LambdaCallback(**hooks_args)

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, callbacks=[test_callback])

    trainer.fit(model)
    trainer.test(model)

    for h in hooks:
        assert h in checker
