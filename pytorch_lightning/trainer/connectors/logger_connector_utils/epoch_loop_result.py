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

from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Union, Tuple, Any

from pytorch_lightning.core.step_result import Result


class LoggerStages(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"


class HookResults:
    """
    This class is used to hold all metrics logged during one callback or model hook.
    Can be used for both training, val, test.

    Result objects will be stored in the following way.

    val and test: self._internals = {"dataloader_idx": []}
    training
        - IF optimizer_idx and training_step_idx are set,
          THEN self._internals = {"dataloader_idx": {"optimizer_idx": {"training_step_idx": [] } } }
        - ELSE self._internals = {"dataloader_idx": []}
    """

    _types = ["list", "dict"]

    def __init__(self, fx_name):
        self._fx_name = fx_name
        self._internals = {}
        self._internals_reduced = {}
        self._internal_type = None
        self.has_reduced = False

    def get_reduced_metrics(self):
        return self._internals_reduced

    def add_dataloader_idx(self):
        return len(self._internals) > 1

    @property
    def num_dataloaders(self):
        return len(self._internals)

    def get_latest_from_dict(self, dl_idx):
        num_opt_idx = len(self._internals[dl_idx]) - 1
        assert num_opt_idx >= 0
        num_opt_idx = str(num_opt_idx)
        num_batch_idx = len(self._internals[dl_idx][num_opt_idx]) - 1
        batch_indexes = [*self._internals[dl_idx][num_opt_idx].keys()]
        # sort them by increasing order
        batch_indexes.sort(key=float)
        assert num_batch_idx >= 0
        return self._internals[dl_idx][num_opt_idx][batch_indexes[-1]][-1]

    def check_dataloader_idx(self, result: Result) -> bool:
        add_dataloader_idx = False
        try:
            if len(result.keys()) > 1:
                random_key = [*result.keys()][-1]
                add_dataloader_idx = result["meta"][random_key]["dataloader_idx"] is not None
                return add_dataloader_idx
            return add_dataloader_idx
        except Exception:
            return add_dataloader_idx

    def get_lastest_from_func_name(self, func_name, *args, latest=True, **kwargs):
        results = {}
        if latest:
            for dl_idx in range(self.num_dataloaders):
                dl_idx = str(dl_idx)
                if self._internal_type == self._types[0]:
                    latest_result = self._internals[dl_idx][-1]
                else:
                    latest_result = self.get_latest_from_dict(dl_idx)
                add_dataloader_idx = self.check_dataloader_idx(latest_result)
                func = getattr(latest_result, func_name)
                results.update(func(*args, add_dataloader_idx=add_dataloader_idx, **kwargs))
            return results
        raise NotImplementedError

    def get_batch_pbar_metrics(self, latest=True, *args, **kwargs):
        return self.get_lastest_from_func_name("get_batch_pbar_metrics", *args, latest=latest, **kwargs)

    def get_batch_log_metrics(self, latest=True, *args, **kwargs):
        return self.get_lastest_from_func_name("get_batch_log_metrics", *args, latest=latest, **kwargs)

    def run_epoch_func(self, results, opt_metric, func_name, *args, **kwargs):
        if isinstance(opt_metric, Result):
            func = getattr(opt_metric, func_name)
            metrics_to_log = func(
                *args,
                add_dataloader_idx=self.add_dataloader_idx,
                **kwargs)
            results.update(metrics_to_log)
        else:
            raise Exception("The provided opt_metric should be a Result Object. Something is wrong")

    def get_epoch_from_func_name(self, func_name, *args, **kwargs):
        results = {}
        for dl_idx in range(self.num_dataloaders):
            dl_idx = str(dl_idx)
            opt_metrics = self._internals_reduced[dl_idx]
            if isinstance(opt_metrics, defaultdict):
                for opt_metric in opt_metrics.values():
                    self.run_epoch_func(results, opt_metric, func_name, *args, **kwargs)
            else:
                self.run_epoch_func(results, opt_metrics, func_name, *args, **kwargs)
        return results

    def get_epoch_pbar_metrics(self, *args, **kwargs):
        return self.get_epoch_from_func_name("get_epoch_pbar_metrics")

    def get_epoch_log_metrics(self, *args, **kwargs):
        return self.get_epoch_from_func_name("get_epoch_log_metrics")

    def get_forked_metrics(self, *args, **kwargs):
        return self.get_epoch_from_func_name("get_forked_metrics")

    @staticmethod
    def _append_to_structure(primary_dict, opt_idx, batch_idx, result):
        if opt_idx not in primary_dict:
            primary_dict[opt_idx] = {}

        if batch_idx not in primary_dict[opt_idx]:
            primary_dict[opt_idx][batch_idx] = []

        primary_dict[opt_idx][batch_idx].append(result)

    def append(self, result, dataloader_idx=None, extra_info: dict = {}):

        assert isinstance(result, Result)

        if dataloader_idx is None:
            dataloader_idx = 0

        primary_key = f"{dataloader_idx}"

        # [dataloader_idx][optimizer_idx][training_step_idx] is a list
        if len(extra_info) > 0:
            self._internal_type = self._types[-1]
            # initialize dictionary
            if primary_key not in self._internals:
                self._internals[primary_key] = {}
                self._internals_reduced[primary_key] = defaultdict(dict)

            # extract infos
            opt_idx = str(extra_info["opt_idx"])
            batch_idx = str(extra_info["batch_idx"])

            self._append_to_structure(self._internals[primary_key], opt_idx, batch_idx, result)

        # [dataloader_idx] is a list
        else:
            self._internal_type = self._types[0]
            if primary_key not in self._internals:
                self._internals[primary_key] = []
            self._internals[primary_key].append(result)

    def auto_reduce_results_on_epoch_end(self):
        if not self.has_reduced:
            epoch_log_metrics = {}
            epoch_progress_bar_metrics = {}

            for dl_idx in range(self.num_dataloaders):
                dl_idx = str(dl_idx)
                epoch_metrics = self._internals[dl_idx]

                if self._internal_type == self._types[-1]:

                    num_opt_idx = len(self._internals[dl_idx]) - 1

                    # Make sure we didn't create key
                    assert num_opt_idx >= 0

                    for opt_idx in range(num_opt_idx + 1):
                        opt_idx = str(opt_idx)
                        opt_outputs = deepcopy(epoch_metrics[opt_idx])

                        num_batch_idx = len(self._internals[dl_idx][str(num_opt_idx)]) - 1
                        assert num_batch_idx >= 0
                        batch_indexes = self._internals[dl_idx][str(num_opt_idx)].keys()

                        # reduce across time first
                        time_reduced_outputs = []
                        for batch_idx in batch_indexes:
                            batch_idx = str(batch_idx)
                            tbptt_outs = opt_outputs[str(batch_idx)]
                            tbptt_outs = tbptt_outs[0].__class__.reduce_across_time(tbptt_outs)
                            if len(tbptt_outs) > 1:
                                time_reduced_outputs.append(tbptt_outs)

                        if len(time_reduced_outputs) == 0:
                            continue

                        # reduce across training steps
                        opt_outputs = time_reduced_outputs[0].__class__.reduce_on_epoch_end(time_reduced_outputs)

                        # with manual opt need 1 + metrics because meta is always there
                        if opt_outputs.minimize is not None:
                            opt_outputs.minimize = opt_outputs.minimize.mean()

                        self._internals_reduced[dl_idx][str(opt_idx)] = opt_outputs
                else:
                    # no need to reduce as called only once
                    if len(epoch_metrics) == 1:
                        reduced_epoch_metrics = deepcopy(epoch_metrics[0])
                    else:
                        reduced_epoch_metrics = epoch_metrics[0].__class__.reduce_on_epoch_end(deepcopy(epoch_metrics))

                    self._internals_reduced[dl_idx] = reduced_epoch_metrics

            self.has_reduced = True

    def __getitem__(self, key: str) -> Any:
        try:
            if key in self._internals:
                return self._internals[key]
            return self[key]
        except KeyError:
            return None

    def __repr__(self):
        return self._internals.__repr__()


class EpochLoopResult:
    """
    This class is responsible to cache all logging metrics which happened during one epoch

    It will cache Result objects as follow.

    self._internals = {"fx_name_0": HookResult(), ..., "fx_name_n": HookResult()}
    """
    def __init__(self, trainer, stage):
        self.trainer = trainer
        self._stage = stage
        self.reset()

    def __getitem__(self, key: str) -> Any:
        try:
            if key in self._internals:
                return self._internals[key]
            return self[key]
        except KeyError:
            return None

    @property
    def has_split_and_opt_idx(self):
        if self._split_idx is not None and self._opt_idx is not None:
            return True
        return False

    @property
    def extra_info(self):
        return {"batch_idx": self.trainer.batch_idx,
                "split_idx": self._split_idx,
                "opt_idx": self._opt_idx}

    def reset_model(self):
        model_ref = self.trainer.get_model()
        model_ref._results = Result()
        model_ref._current_hook_fx_name = ''
        model_ref._current_fx_name = ''

    def current_model_info(self):
        model_ref = self.trainer.get_model()
        # extract hook information
        fx_name = model_ref._current_hook_fx_name
        if fx_name == '':
            fx_name = model_ref._current_fx_name
        dataloader_idx = model_ref._current_dataloader_idx
        return fx_name, dataloader_idx

    def cache_result(self):
        model_ref = self.trainer.get_model()

        # extract hook results
        hook_result = model_ref._results

        # extract model information
        fx_name, dataloader_idx = self.current_model_info()

        # add only if anything as been logged
        # default len is 1 due to _internals
        if len(hook_result) > 1:

            if fx_name not in self._internals:
                self._internals[fx_name] = HookResults(fx_name)

            extra_info = {}
            if self.has_split_and_opt_idx:
                extra_info = self.extra_info

            # attach capture batch_size
            Result.attach_batch_size(self._batch_size, hook_result)

            self._internals[fx_name].append(
                deepcopy(hook_result),
                dataloader_idx=dataloader_idx,
                extra_info=extra_info)

            # update logged_metrics, progress_bar_metrics, callback_metrics
            self.update_logger_connector()

        # reset _results, fx_name
        self.reset_model()

    def update_logger_connector(self):
        """
        This function is called every time we capture a hook
        It automatically updates the logger_connector followings:
            -  progress_bar_metrics with pbar_metrics
            -  logged_metrics with log_metrics
            -  callback_metrics with progress_bar_metrics + logged_metrics
        """

        logger_connector = self.trainer.logger_connector

        callback_metrics = {}

        if not self._has_batch_loop_finished:
            # get pbar
            batch_pbar_metrics = self.get_latest_batch_pbar_metrics()
            logger_connector.add_progress_bar_metrics(batch_pbar_metrics)

            # get logged_metrics
            batch_log_metrics = self.get_latest_batch_log_metrics()
            logger_connector.logged_metrics.update(batch_log_metrics)

            callback_metrics.update(batch_pbar_metrics)
            callback_metrics.update(batch_log_metrics)
        else:
            epoch_dict = {"epoch": self.trainer.current_epoch}

            # get pbar
            epoch_pbar_metrics = self.get_epoch_pbar_metrics()
            logger_connector.add_progress_bar_metrics(epoch_pbar_metrics)

            # get logged_metrics
            epoch_log_metrics = self.get_epoch_log_metrics()
            logger_connector.logged_metrics.update(epoch_log_metrics)
            logger_connector.logged_metrics.update(epoch_dict)

            # get forked_metrics
            forked_metrics = self.get_forked_metrics()

            callback_metrics.update(epoch_pbar_metrics)
            callback_metrics.update(epoch_log_metrics)
            callback_metrics.update(forked_metrics)

        # update callback_metrics
        logger_connector.callback_metrics.update(callback_metrics)
        logger_connector.callback_metrics.pop("epoch", None)

    def get_latest_batch_log_metrics(self):
        results = {}
        for fx_name, hook_result in self._internals.items():
            results.update(hook_result.get_batch_log_metrics(
                latest=True,
                include_forked_originals=False))
        return results

    def get_latest_batch_pbar_metrics(self):
        results = {}
        for fx_name, hook_result in self._internals.items():
            results.update(hook_result.get_batch_pbar_metrics(
                latest=True,
                include_forked_originals=False))
        return results

    @property
    def has_reduced(self):
        hook_results = self._internals.values()
        return len(hook_results) == sum([h.has_reduced for h in hook_results])

    def auto_reduce_results_on_epoch_end(self):
        if not self.has_reduced:
            for fx_name, hook_result in self._internals.items():
                hook_result.auto_reduce_results_on_epoch_end()

    @property
    def has_batch_loop_finished(self):
        return self._has_batch_loop_finished

    @has_batch_loop_finished.setter
    def has_batch_loop_finished(self, has_batch_loop_finished):
        if has_batch_loop_finished:
            # If batch loop has finished, reduce metrics
            self.auto_reduce_results_on_epoch_end()

            # batch_size should be none as we finished batch loop
            self._batch_size = None

        self._has_batch_loop_finished = has_batch_loop_finished
        self.update_logger_connector()

    def get_epoch_pbar_metrics(self):
        if not self.has_reduced:
            self.auto_reduce_results_on_epoch_end()
        epoch_pbar_metrics = {}
        for fx_name, hook_result in self._internals.items():
            epoch_pbar_metrics.update(hook_result.get_epoch_pbar_metrics())
        return epoch_pbar_metrics

    def get_epoch_log_metrics(self):
        if not self.has_reduced:
            self.auto_reduce_results_on_epoch_end()
        epoch_log_metrics = {}
        for fx_name, hook_result in self._internals.items():
            epoch_log_metrics.update(hook_result.get_epoch_log_metrics())
        return epoch_log_metrics

    def get_forked_metrics(self):
        if not self.has_reduced:
            self.auto_reduce_results_on_epoch_end()
        forked_metrics = {}
        for fx_name, hook_result in self._internals.items():
            forked_metrics.update(hook_result.get_forked_metrics())
        return forked_metrics

    def get_reduced_metrics(self):
        if not self.has_reduced:
            self.auto_reduce_results_on_epoch_end()
        reduced_metrics = {}
        for fx_name, hook_result in self._internals.items():
            reduced_metrics[fx_name] = hook_result.get_reduced_metrics()
        return reduced_metrics

    def __repr__(self):
        return f"{self.__class__.__name__}(stage={self._stage}, internals={self._internals})"

    def reset(self):
        self._internals = {}
        self._dataloader_idx = None
        self._split_idx = None
        self._opt_idx = None
        self._batch_size = None
        self._has_batch_loop_finished = False
        self._num_dataloaders = 1
