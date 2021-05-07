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
""" Test deprecated functionality which will be removed in v1.6.0 """

import pytest

from pytorch_lightning import Trainer
from tests.helpers import BoringModel


def test_v1_6_0_trainer_model_hook_mixin(tmpdir):
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, checkpoint_callback=False, logger=False)
    trainer.fit(model)
    with pytest.deprecated_call(match="is deprecated in v1.4 and will be removed in v1.6"):
        trainer.is_function_implemented("training_step", model)

    with pytest.deprecated_call(match="is deprecated in v1.4 and will be removed in v1.6"):
        trainer.has_arg("training_step", "batch")
