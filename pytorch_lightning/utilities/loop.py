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
from collections import defaultdict, ChainMap


class CacheInternalMetrics:
    """
    This class is an helper to cache model._results logged values before / after entering batch loop.
    As on every `run_training_batch`, we apply model._results = Result()
    and therefore delete any previously logged values

    before_on_batch_start is responsible to catch logging values from `on_start` to `on_batch_start`
    after_on_batch_end is responsible to catch logging values from `on_batch_end` to `on_epoch_end`
    """

    stages = ["before_on_batch_start", "after_on_batch_end"]

    def __init__(self):
        self._internal_dict = {stage: defaultdict(list) for stage in self.stages}

    def append(self, stage: str, key: str, value) -> None:
        assert stage in self.stages, f"Provided stage {stage} should be within {self.stages}"
        self._internal_dict[stage][key].append(value)

    def get_as_dict(self, stage, key):
        _internal_metrics = self.get_as_list(stage, key)
        return dict(ChainMap(*_internal_metrics))

    def get_as_list(self, stage, key):
        assert stage in self.stages, f"Provided stage {stage} should be within {self.stages}"
        return self._internal_dict[stage][key]

    def __repr__(self):
        return self._internal_dict.__repr__()

    def update(self, trainer, stage: str) -> None:
        """
        This function is used to cache any logged information
        between "on_train_start" to "on_train_epoch_start" callback hooks
        """
        assert stage in self.stages, f"Provided stage {stage} should be within {self.stages}"
        if not trainer.running_sanity_check:
            model_ref = trainer.get_model()

            # save epoch metrics
            self.append(stage, "epoch_log_metrics", model_ref._results.get_epoch_log_metrics())
            self.append(stage, "epoch_pbar_metrics", model_ref._results.get_epoch_pbar_metrics())

            # save step/batch metrics
            self.append(stage, "batch_log_metrics", model_ref._results.get_batch_log_metrics())
            self.append(stage, "batch_pbar_metrics", model_ref._results.get_batch_pbar_metrics())
