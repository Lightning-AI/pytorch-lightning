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
from typing import Optional, Union

from pytorch_lightning.utilities import GradClipAlgorithmType, rank_zero_deprecation
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class TrainingTricksConnector:
    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(
        self,
        gradient_clip_val: Optional[Union[int, float]],
        gradient_clip_algorithm: Optional[str],
        track_grad_norm: Union[int, float, str],
        terminate_on_nan: Optional[bool],
    ):
        if terminate_on_nan is not None:
            rank_zero_deprecation(
                "Trainer argument `terminate_on_nan` was deprecated in v1.5 and will be removed in 1.7."
                " Please use `Trainer(detect_anomaly=True)` instead."
            )
            if not isinstance(terminate_on_nan, bool):
                raise TypeError(f"`terminate_on_nan` should be a bool, got {terminate_on_nan}.")

        # gradient clipping
        if gradient_clip_val is not None and not isinstance(gradient_clip_val, (int, float)):
            raise TypeError(f"`gradient_clip_val` should be an int or a float. Got {gradient_clip_val}.")

        if gradient_clip_algorithm is not None and not GradClipAlgorithmType.supported_type(
            gradient_clip_algorithm.lower()
        ):
            raise MisconfigurationException(
                f"`gradient_clip_algorithm` {gradient_clip_algorithm} is invalid. "
                f"Allowed algorithms: {GradClipAlgorithmType.supported_types()}."
            )

        # gradient norm tracking
        if track_grad_norm != -1 and not (
            (isinstance(track_grad_norm, (int, float)) or track_grad_norm == "inf") and float(track_grad_norm) > 0
        ):
            raise MisconfigurationException(
                f"`track_grad_norm` must be a positive number or 'inf' (infinity norm). Got {track_grad_norm}."
            )

        self.trainer._terminate_on_nan = terminate_on_nan
        self.trainer.gradient_clip_val = gradient_clip_val
        self.trainer.gradient_clip_algorithm = (
            GradClipAlgorithmType(gradient_clip_algorithm.lower())
            if gradient_clip_algorithm is not None
            else gradient_clip_algorithm
        )
        self.trainer.track_grad_norm = float(track_grad_norm)
