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
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.callbacks import GradientAccumulationScheduler


class TrainingTricksConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(
            self,
            gradient_clip_val,
            track_grad_norm,
            accumulate_grad_batches,
            truncated_bptt_steps,
            terminate_on_nan
    ):

        self.trainer.terminate_on_nan = terminate_on_nan

        # gradient clipping
        self.trainer.gradient_clip_val = gradient_clip_val

        # gradient norm tracking
        if not isinstance(track_grad_norm, (int, float)) and track_grad_norm != 'inf':
            raise MisconfigurationException("track_grad_norm can be an int, a float or 'inf' (infinity norm).")
        self.trainer.track_grad_norm = float(track_grad_norm)

        # accumulated grads
        self.trainer.accumulate_grad_batches = accumulate_grad_batches
        self.configure_accumulated_gradients(accumulate_grad_batches)

        self.trainer.truncated_bptt_steps = truncated_bptt_steps

    def configure_accumulated_gradients(self, accumulate_grad_batches):
        if isinstance(accumulate_grad_batches, dict):
            self.trainer.accumulation_scheduler = GradientAccumulationScheduler(accumulate_grad_batches)
        elif isinstance(accumulate_grad_batches, int):
            schedule = {0: accumulate_grad_batches}
            self.trainer.accumulation_scheduler = GradientAccumulationScheduler(schedule)
        else:
            raise TypeError("Gradient accumulation supports only int and dict types")
