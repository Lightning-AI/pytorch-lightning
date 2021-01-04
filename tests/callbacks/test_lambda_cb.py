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
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LambdaCallback
from tests.base.boring_model import BoringModel


def test_lambda_call(tmpdir):
    seed_everything(42)

    model = BoringModel()
    checker = set()

    callback_dicts = {"setup": lambda *args: checker.add("setup")}
    test_callback = LambdaCallback(**callback_dicts)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=1,
        limit_train_batches=3,
        limit_test_batches=2,
        progress_bar_refresh_rate=0,
        enable_pl_optimizer=True,
        callbacks=[test_callback],
    )

    trainer.fit(model)

    for name in ("setup",):
        assert name in checker
