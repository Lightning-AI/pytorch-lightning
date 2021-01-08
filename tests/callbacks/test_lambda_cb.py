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
from tests.base.boring_model import BoringModel


def test_lambda_call(tmpdir):
    seed_everything(42)

    checker = set()

    hooks = [m for m, _ in inspect.getmembers(Callback, predicate=inspect.isfunction)]
    model = BoringModel()

    hooks_args = {h: (lambda x: lambda *args: checker.add(x))(h) for h in hooks}
    test_callback = LambdaCallback(**hooks_args)

    trainer = Trainer(
        default_root_dir=tmpdir,
        num_sanity_val_steps=1,
        max_epochs=1,
        callbacks=[test_callback]
    )
    results = trainer.fit(model)
    trainer.test(model)

    assert results
    assert checker == set(hooks)
