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
from abc import ABC
from typing import Any, Dict, List, Optional, Union


class TrainerAttributes(ABC):
    # This class represents the types of attributes set dynamically on the Trainer class.
    # Any attribute set dynamically on trainer can be added here to aid MyPy type checking.
    # A comment next to the attribute indicates where they are set or referenced.

    should_stop: bool  # callbacks/early_stopping.py
    accumulate_grad_batches: Optional[int]  # callbacks/gradient_accumulation_scheduler.py
    gpus: Optional[Union[List[int], str, int]]  # accelerators/accelerator_connector.py
    log_every_n_steps: int  # connectors/logger_connector/logger_connector.py
    lr_schedulers: List[Dict[str, Any]]  # accelerator/accelerator.py
    batch_idx: int  # trainer/training_loop.py
    check_val_every_n_epoch: int  # trainer/connectors/data_connector.py
