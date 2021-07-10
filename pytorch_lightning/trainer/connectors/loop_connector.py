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
from pytorch_lightning.loops.base import Loop


class LoopConnector(object):

    def __init__(self, trainer: "pl.Trainer"):
        self.trainer = trainer

    def on_trainer_init(self, *args, **kwargs):
        self.trainer.fit_loop = self.on_fit_loop_init(*args, **kwargs)
        self.trainer.validate_loop = self.on_validate_loop_init(*args, **kwargs)
        self.trainer.test_loop = self.on_test_loop_init(*args, **kwargs)
        self.trainer.predict_loop = self.on_predict_loop_init(*args, **kwargs)

        self.connect(self.trainer)

    def on_fit_loop_init(
        self,
        min_epochs: Optional[int] = None,
        max_epochs: Optional[int] = None,
        min_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        **__,
    ) -> Loop:
        return FitLoop(min_epochs=min_epochs, max_epochs=max_epochs, min_steps=min_steps, max_steps=max_steps)

    def on_validate_loop_init(self, *_, **__) -> Loop:
        return EvaluationLoop()

    def on_test_loop_init(self, *_, **__) -> Loop:
        return EvaluationLoop()

    def on_predict_loop_init(self, *_, **__) -> Loop:
        return PredictionLoop()

    def connect(self, trainer: 'pl.Trainer') -> None:
        trainer.fit_loop.trainer = trainer
        trainer.validate_loop.trainer = trainer
        trainer.test_loop.trainer = trainer
        trainer.predict_loop.trainer = trainer

    def load_state_dict(self, state_dict):
        self.trainer.fit_loop.load_state_dict(state_dict["fit_loop"])
        self.trainer.validate_loop.load_state_dict(state_dict["validate_loop"])
        self.trainer.test_loop.load_state_dict(state_dict["test_loop"])
        self.trainer.predict_loop.load_state_dict(state_dict["predict_loop"])
