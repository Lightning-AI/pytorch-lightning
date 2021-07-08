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
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.loops import EvaluationLoop, FitLoop, PredictionLoop


class LoopConnector(object):

    def __init__(self, trainer: "pl.Trainer"):
        self.trainer = trainer

    def on_trainer_init(self, *args, **kwargs):
        self.trainer.fit_loop = self.on_fit_loop_init(*args, **kwargs)
        self.trainer.validate_loop = self.on_validate_loop_init(*args, **kwargs)
        self.trainer.test_loop = self.on_test_loop_init(*args, **kwargs)
        self.trainer.predict_loop = self.on_predict_loop_init(*args, **kwargs)

        self.connect_loops_to_trainer()

    def on_fit_loop_init(
        self,
        min_epochs: Optional[int] = None,
        max_epochs: Optional[int] = None,
        min_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        **__,
    ):
        return FitLoop(min_epochs=min_epochs, max_epochs=max_epochs, min_steps=min_steps, max_steps=max_steps)

    def on_validate_loop_init(self, *_, **__):
        return EvaluationLoop()

    def on_test_loop_init(self, *_, **__):
        return EvaluationLoop()

    def on_predict_loop_init(self, *_, **__):
        return PredictionLoop()

    def connect_loops_to_trainer(self):
        self.trainer.fit_loop.trainer = self.trainer
        self.trainer.validate_loop.trainer = self.trainer
        self.trainer.test_loop.trainer = self.trainer
        self.trainer.predict_loop.trainer = self.trainer

        breakpoint()
