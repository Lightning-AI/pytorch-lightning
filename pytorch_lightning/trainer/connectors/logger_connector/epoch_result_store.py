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
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch

from pytorch_lightning.core.step_result import Result


class LoggerStages(str, Enum):
    """ Train/validation/test phase in each training step.

    >>> # you can math the type with string
    >>> LoggerStages.TRAIN == 'train'
    True
    """
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"

    @staticmethod
    def determine_stage(stage_or_testing: Union[str, bool]) -> 'LoggerStages':
        if isinstance(stage_or_testing, str) and stage_or_testing in list(LoggerStages):
            return LoggerStages(stage_or_testing)
        if isinstance(stage_or_testing, (bool, int)):
            # stage_or_testing is trainer.testing
            return LoggerStages.TEST if bool(stage_or_testing) else LoggerStages.VAL
        raise RuntimeError(f"Invalid stage {stage_or_testing} of type {type(stage_or_testing)} given")


class ResultStoreType(str, Enum):
    INSIDE_BATCH_TRAIN_LOOP = "inside_batch_train_loop"
    OUTSIDE_BATCH_TRAIN_LOOP = "outside_batch_train_loop"


class HookResultStore:
    """
    This class is defined for internal usage.
    It holds all metrics logged using the self.log function
    in the scope of ModelHooks or Callback functions.

    We need to differentiate 3 different scenarios:
        - (1): We are outside of a batch loop
            * It means no dataloader_idx, no optimizer idx, etc..
        - (2): We are inside the training batch loop
            * We have an optimizer idx and split idx to track
        - (3): We are inside the evaluation loop
            * We have a dataloader_idx to track

    The data store `Result` objects for those 3 scenarios in `self._internals`.

    (1): self._internals = {dataloader_idx: [Result(), ..., Result()]}
        * dataloader_idx not being defined, it is set to 0 b default
    (2): self._internals = {dataloader_idx: {optimizer_idx: {batch_idx: [Result(), ..., Result()]}}}
    (3): Same as (1) for simplicity

    Those data structures enables us to reduce properly Result object when batch loop is finished.
    """

    def __init__(self, fx_name):
        self._fx_name = fx_name
        self._internals = {}
        self._internals_reduced = {}
        self._internal_type = None
        self.has_reduced = False
        self._latest_ref = {}

    @property
    def has_several_dataloaders(self) -> bool:
        return self.num_dataloaders > 1

    @property
    def num_dataloaders(self) -> int:
        inter = self._internals_reduced if self.has_reduced else self._internals
        return len(inter)

    def check_dataloader_idx(self, result: Result) -> bool:
        random_key = list(result.keys())[-1]
        return result["meta"][random_key]["dataloader_idx"] is not None

    def get_latest_from_func_name(self, latest_result_opt, func_name: str, *args, **kwargs) -> Dict:
        results = {}
        for opt_idx in latest_result_opt:
            latest_result = latest_result_opt[opt_idx]
            add_dataloader_idx = self.check_dataloader_idx(latest_result)
            func = getattr(latest_result, func_name)
            results.update(func(*args, add_dataloader_idx=add_dataloader_idx, **kwargs))
        return results

    def run_latest_batch_metrics_with_func_name(self, func_name, *args, **kwargs) -> List[Dict]:
        """
        This function used cache_ref and cache_result to optimize loading metrics

        Context: As we update the logger_connector metrics on every `self.log` call,
        and it can be pretty time consuming, especially when logging outside batch loop.

        HookResultStore keeps track of its latest added result object,
        and cache its pbar and log metrics if already called on,
        """
        return [
            self.get_latest_from_func_name(self._latest_ref[dl_idx], func_name, *args, **kwargs)
            for dl_idx in range(self.num_dataloaders)
        ]

    def get_batch_pbar_metrics(self, *args, **kwargs):
        return self.run_latest_batch_metrics_with_func_name("get_batch_pbar_metrics", *args, **kwargs)

    def get_batch_log_metrics(self, *args, **kwargs):
        return self.run_latest_batch_metrics_with_func_name("get_batch_log_metrics", *args, **kwargs)

    def run_epoch_func(self, results, opt_metric, func_name, *args, **kwargs) -> None:
        if not isinstance(opt_metric, Result):
            raise Exception("The provided opt_metric should be a Result Object. Something is wrong")
        func = getattr(opt_metric, func_name)
        metrics_to_log = func(*args, add_dataloader_idx=self.has_several_dataloaders, **kwargs)
        results.append(metrics_to_log)

    def get_epoch_from_func_name(self, func_name, *args, **kwargs) -> List[Dict]:
        results = []
        for dl_idx in range(self.num_dataloaders):
            opt_metrics = self._internals_reduced[dl_idx]
            if isinstance(opt_metrics, defaultdict):
                for opt_metric in opt_metrics.values():
                    self.run_epoch_func(results, opt_metric, func_name, *args, **kwargs)
            else:
                self.run_epoch_func(results, opt_metrics, func_name, *args, **kwargs)
        return results

    def get_epoch_pbar_metrics(self, *_, **__) -> List[Dict]:
        return self.get_epoch_from_func_name("get_epoch_pbar_metrics")

    def get_epoch_log_metrics(self, *_, **__) -> List[Dict]:
        return self.get_epoch_from_func_name("get_epoch_log_metrics")

    def get_forked_metrics(self, *_, **__) -> List[Dict]:
        return self.get_epoch_from_func_name("get_forked_metrics")

    @staticmethod
    def _append_to_structure(primary_dict, opt_idx, batch_idx, result) -> None:
        primary_dict.setdefault(opt_idx, {})
        primary_dict[opt_idx].setdefault(batch_idx, [])
        primary_dict[opt_idx][batch_idx].append(result)

    def append(self, result, dataloader_idx: Optional[int] = None, extra_info: Optional[dict] = None) -> None:
        assert isinstance(result, Result)
        if dataloader_idx is None:
            dataloader_idx = 0

        if extra_info is None:
            extra_info = {}

        # [dataloader_idx][optimizer_idx][training_step_idx] is a list
        if len(extra_info) > 0:
            self._internal_type = ResultStoreType.INSIDE_BATCH_TRAIN_LOOP
            # initialize dictionary
            if dataloader_idx not in self._internals:
                self._internals[dataloader_idx] = {}
                self._internals_reduced[dataloader_idx] = defaultdict(dict)
                self._latest_ref[dataloader_idx] = {}

            # extract infos
            opt_idx = extra_info["opt_idx"]
            batch_idx = extra_info["batch_idx"]

            self._append_to_structure(self._internals[dataloader_idx], opt_idx, batch_idx, result)

            self._latest_ref[dataloader_idx][opt_idx] = result

        # [dataloader_idx] is a list
        else:
            self._internal_type = ResultStoreType.OUTSIDE_BATCH_TRAIN_LOOP
            self._internals.setdefault(dataloader_idx, [])
            self._internals[dataloader_idx].append(result)

            if dataloader_idx not in self._latest_ref:
                self._latest_ref[dataloader_idx] = {}
                self._latest_ref[dataloader_idx][0] = {}

            self._latest_ref[dataloader_idx][0] = result

    def auto_reduce_results_on_epoch_end(self) -> None:
        """
        This function is called to reduce `self._internals` Result object.
        The reduced Result object will be saved into `self._internals_reduced`
        The `self._internals` stored Result objects will be deleted to save memory.
        """
        if self.has_reduced:
            return
        for dl_idx in range(self.num_dataloaders):
            epoch_metrics = self._internals[dl_idx]

            if self._internal_type == ResultStoreType.INSIDE_BATCH_TRAIN_LOOP:
                for opt_idx in list(epoch_metrics):
                    # TODO: Figure out to reduce memory
                    # TODO: How to start training in middle of epoch
                    opt_outputs = epoch_metrics[opt_idx]

                    # reduce across time first
                    time_reduced_outputs = []
                    for batch_idx in opt_outputs.keys():
                        tbptt_outs = opt_outputs[batch_idx]
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

                    self._internals_reduced[dl_idx][opt_idx] = opt_outputs

                    # free memory
                    del self._internals[dl_idx][opt_idx]
            else:
                # no need to reduce as called only once
                if len(epoch_metrics) == 1:
                    reduced_epoch_metrics = epoch_metrics[0]
                else:
                    reduced_epoch_metrics = epoch_metrics[0].__class__.reduce_on_epoch_end(epoch_metrics)

                self._internals_reduced[dl_idx] = reduced_epoch_metrics

                # free memory
                del self._internals[dl_idx]

        self.has_reduced = True

    def __getitem__(self, key: str) -> Any:
        return self._internals.get(key, None)

    def __repr__(self):
        return self._internals.__repr__()


class EpochResultStore:
    """
    This class is defined for internal usage.
    It holds all metrics logged using the self.log function using `HookResultStore` object.
    The internal datastructure is as follow:
    self._internals = {"fx_name_0": HookResultStore(), ..., "fx_name_n": HookResultStore()}
    Pseudo Code Example:
    ```
    model._current_fx_name = 'something'
    model._results = Result()
    model.log('a', ...)
    epoch_result_store.cache_result()
    ```
    """

    def __init__(self, trainer, stage):
        self.trainer = trainer
        self._stage = stage
        self.reset()

    def __getitem__(self, key: str) -> Any:
        return self._internals.get(key, None)

    @property
    def has_split_and_opt_idx(self):
        """
        This function informs if we are running within training batch loop
        """
        return self._split_idx is not None and self._opt_idx is not None

    @property
    def extra_info(self):
        """
        This function provides necessary parameters to properly configure HookResultStore obj
        """
        return {"batch_idx": self.trainer.batch_idx, "split_idx": self._split_idx, "opt_idx": self._opt_idx}

    def reset_model(self):
        """
        This function is used to reset model state at the end of the capture
        """
        model_ref = self.trainer.get_model()
        model_ref._results = Result()
        model_ref._current_hook_fx_name = None
        model_ref._current_fx_name = ''

    def current_model_info(self):
        """
        This function is used to extract
        information related to current function scoping `self.log` call.
        """
        model_ref = self.trainer.get_model()
        # extract hook information
        fx_name = model_ref._current_hook_fx_name or model_ref._current_fx_name
        dataloader_idx = model_ref._current_dataloader_idx
        return fx_name, dataloader_idx

    def cache_result(self) -> None:
        """
        This function is called after every hook
        and store the result object
        """
        with self.trainer.profiler.profile("cache_result"):
            model_ref = self.trainer.get_model()

            # extract hook results
            hook_result = model_ref._results

            if len(hook_result) == 1:
                model_ref._current_hook_fx_name = None
                model_ref._current_fx_name = ''
                return

            # extract model information
            fx_name, dataloader_idx = self.current_model_info()

            self._internals.setdefault(fx_name, HookResultStore(fx_name))

            extra_info = self.extra_info if self.has_split_and_opt_idx else {}

            # attach capture batch_size
            Result.attach_batch_size(self._batch_size, hook_result)

            hook_result.detach()
            if self.trainer.move_metrics_to_cpu:
                hook_result.cpu()
            elif self.trainer.use_dp:
                hook_result.to(torch.device("cuda", self.trainer.root_gpu))

            self._internals[fx_name].append(hook_result, dataloader_idx=dataloader_idx, extra_info=extra_info)

            # update logged_metrics, progress_bar_metrics, callback_metrics

            if "epoch_end" in fx_name:
                self.update_logger_connector()

            self.reset_model()

    def update_logger_connector(self) -> None:
        """
        This function is called every time we capture a hook
        It automatically updates the logger_connector followings:
            -  progress_bar_metrics with pbar_metrics
            -  logged_metrics with log_metrics
            -  callback_metrics with progress_bar_metrics + logged_metrics
        """

        logger_connector = self.trainer.logger_connector

        callback_metrics = {}
        batch_pbar_metrics = {}
        batch_log_metrics = {}
        is_train = self._stage in LoggerStages.TRAIN.value

        if not self._has_batch_loop_finished:
            # get pbar
            batch_pbar_metrics = self.get_latest_batch_pbar_metrics()
            logger_connector.add_progress_bar_metrics(batch_pbar_metrics)
            batch_log_metrics = self.get_latest_batch_log_metrics()

            if is_train:
                # Only log and add to callback epoch step during evaluation, test.
                logger_connector.logged_metrics.update(batch_log_metrics)
                callback_metrics.update(batch_pbar_metrics)
                callback_metrics.update(batch_log_metrics)
        else:
            # get pbar
            epoch_pbar_metrics = self.get_epoch_pbar_metrics()
            logger_connector.add_progress_bar_metrics(epoch_pbar_metrics)

            # get logged_metrics
            epoch_log_metrics = self.get_epoch_log_metrics()
            logger_connector.logged_metrics.update(epoch_log_metrics)
            logger_connector.logged_metrics.update(epoch=self.trainer.current_epoch)

            # get forked_metrics
            forked_metrics = self.get_forked_metrics()

            callback_metrics.update(epoch_pbar_metrics)
            callback_metrics.update(epoch_log_metrics)
            callback_metrics.update(forked_metrics)

        if not is_train and self.trainer.testing:
            logger_connector.evaluation_callback_metrics.update(callback_metrics)

        # update callback_metrics
        logger_connector.callback_metrics.update(callback_metrics)
        logger_connector.callback_metrics.pop("epoch", None)

        batch_pbar_metrics.pop("debug_epoch", None)
        return batch_pbar_metrics, batch_log_metrics

    def run_batch_from_func_name(self, func_name) -> Dict:
        results = [getattr(hook_result, func_name) for hook_result in self._internals.values()]
        results = [func(include_forked_originals=False) for func in results]
        return {k: v for d in sum(results, []) for k, v in d.items()}  # List[List[dict]] -> dict

    def get_latest_batch_log_metrics(self) -> Dict:
        batch_log_metrics = self.run_batch_from_func_name("get_batch_log_metrics")
        batch_log_metrics.update(self.legacy_batch_log_metrics)
        return batch_log_metrics

    def get_latest_batch_pbar_metrics(self) -> Dict:
        batch_pbar_metrics = self.run_batch_from_func_name("get_batch_pbar_metrics")
        batch_pbar_metrics.update(self.legacy_batch_pbar_metrics)
        return batch_pbar_metrics

    @property
    def has_reduced(self) -> bool:
        hook_results = self._internals.values()
        return len(hook_results) == sum(h.has_reduced for h in hook_results)

    def auto_reduce_results_on_epoch_end(self) -> None:
        if not self.has_reduced:
            for hook_result in self._internals.values():
                hook_result.auto_reduce_results_on_epoch_end()

    @property
    def has_batch_loop_finished(self) -> bool:
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

    def run_epoch_by_func_name(self, func_name) -> Dict:
        if not self.has_reduced:
            self.auto_reduce_results_on_epoch_end()
        results = [getattr(hook_result, func_name) for hook_result in self._internals.values()]
        results = [func() for func in results]
        return {k: v for d in sum(results, []) for k, v in d.items()}  # List[List[dict]] -> dict

    def get_epoch_pbar_metrics(self) -> Dict:
        return self.run_epoch_by_func_name("get_epoch_pbar_metrics")

    def get_epoch_log_metrics(self) -> Dict:
        return self.run_epoch_by_func_name("get_epoch_log_metrics")

    def get_forked_metrics(self) -> Dict:
        return self.run_epoch_by_func_name("get_forked_metrics")

    def reset(self):
        self._internals = {}
        self._dataloader_idx: Optional[int] = None
        self._split_idx: Optional[int] = None
        self._opt_idx: Optional[int] = None
        self._batch_size: Optional[int] = None
        self._has_batch_loop_finished = False
        self.legacy_batch_log_metrics = {}
        self.legacy_batch_pbar_metrics = {}

    def __call__(
        self,
        fx_name: str,
        dl_idx: Optional[int] = None,
        opt_idx: Optional[int] = None,
        batch_idx: Optional[int] = None,
        split_idx: Optional[int] = None,
        reduced: bool = False,
    ):
        """
        This function is an helper to access stored data

        It access data from the HookResultStore. Please,
        check its data structure for better understanding

        Data can be accessed with the following chains:

        IF REDUCED:
            * IF accessing a fx_name defined in batch training loop:
                fx_name -> dl_idx -> opt_idx -> batch_idx -> split_idx
            * ELSE fx_name -> dl_idx -> batch_idx
        ELSE:
            * IF accessing a fx_name defined in batch training loop:
                fx_name -> dl_idx -> opt_idx
            * ELSE fx_name -> dl_idx

        Note:
            As soon as a param is None, it breaks the chain and returns associated stored data.

        Example::

            result: Result = self(fx_name="training_step", dl_idx="0", opt_idx="0", reduced=True)
            result['train_loss_epoch'] # aggregated train_loss over one epoch.

        Args:

            fx_name: Hook name from ModelHooks or Callback. Example: `training_step`

            dl_idx: Dataloader idx in short. It starts from 0 to num_dataloaders - 1

            opt_idx: Optimizer idx in short. It starts from 0 to num_optimizers - 1

            batch_idx: Index of batch idx seen during batch training or evaluation.
                Works only with reduced=False

            split_idx: Index of split idx in training loop when ttbt is used.

            reduced: Data are being aggregated on on_epoch_end.
                Indicates if we want to access aggregated Result or not.
        """
        hook_result = self[fx_name]
        internal_type = hook_result._internal_type
        result = hook_result._internals_reduced if reduced else hook_result._internals

        if dl_idx is not None:
            result = result[dl_idx]
            if internal_type == ResultStoreType.INSIDE_BATCH_TRAIN_LOOP:
                if opt_idx is not None:
                    result = result[opt_idx]
                    if not reduced and batch_idx is not None:
                        result = result[batch_idx]
                        if split_idx is not None:
                            result = result[split_idx]
            elif not reduced and batch_idx is not None:
                result = result[batch_idx]
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(stage={self._stage}, internals={self._internals})"
