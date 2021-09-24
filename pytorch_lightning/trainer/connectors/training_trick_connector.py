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
from typing import Dict, Union

from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class TrainingTricksConnector:
    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(
        self,
        gradient_clip_val: Union[int, float],
        gradient_clip_algorithm: str,
        track_grad_norm: Union[int, float, str],
        accumulate_grad_batches: Union[int, Dict[int, int]],
        terminate_on_nan: bool,
    ):
        if not isinstance(terminate_on_nan, bool):
            raise MisconfigurationException(f"`terminate_on_nan` should be a bool, got {terminate_on_nan}.")

        # gradient clipping
        if not isinstance(gradient_clip_val, (int, float)):
            raise MisconfigurationException(
                f"Gradient Clipping Value should be an int or a float. Got {gradient_clip_val}."
            )

        if not GradClipAlgorithmType.supported_type(gradient_clip_algorithm.lower()):
            raise MisconfigurationException(
                f"Gradient Clipping Algorithm {gradient_clip_algorithm} is invalid. "
                f"Allowed algorithms: {GradClipAlgorithmType.supported_types()}."
            )

        # gradient norm tracking
        if not isinstance(track_grad_norm, (int, float)) and track_grad_norm != "inf":
            raise MisconfigurationException(
                f"`track_grad_norm` should be an int, a float or 'inf' (infinity norm). Got {track_grad_norm}."
            )

        self.trainer.terminate_on_nan = terminate_on_nan
        self.trainer.gradient_clip_val = gradient_clip_val
        self.trainer.gradient_clip_algorithm = GradClipAlgorithmType(gradient_clip_algorithm.lower())
        self.trainer.track_grad_norm = float(track_grad_norm)

        # accumulated grads
        self.trainer.accumulate_grad_batches = accumulate_grad_batches
        self.configure_accumulated_gradients(accumulate_grad_batches)

    def configure_accumulated_gradients(self, accumulate_grad_batches: Union[int, Dict[int, int]]) -> None:
        if isinstance(accumulate_grad_batches, dict):
            self.trainer.accumulation_scheduler = GradientAccumulationScheduler(accumulate_grad_batches)
        elif isinstance(accumulate_grad_batches, int):
            schedule = {0: accumulate_grad_batches}
            self.trainer.accumulation_scheduler = GradientAccumulationScheduler(schedule)
        else:
            raise TypeError("Gradient accumulation supports only int and dict types")
