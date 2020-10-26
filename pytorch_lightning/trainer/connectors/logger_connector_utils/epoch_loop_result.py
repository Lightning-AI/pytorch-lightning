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

from pytorch_lightning.core.step_result import Result

class LoggerStages(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"

class HookResults:

    """
    This class is used to hold all metrics logged during one callback or model hook. 
    Can be used for both training and val, test
    """

    _types = ["list", "dict"]

    def __init__(self, fx_name):
        self._fx_name = fx_name
        self._internals = {}
        self._internals_reduced = {}
        self._internal_type = None
        self.has_reduced = False
    
    @property
    def num_dataloaders(self):
        return len(self._internals)

    @property
    def add_dataloader_idx(self):
        return True if self.num_dataloaders > 1 else False

    def get_latest_from_dict(self, dl_idx):
        num_opt_idx = len(self._internals[dl_idx]) - 1
        assert num_opt_idx >= 0
        num_opt_idx = str(num_opt_idx)
        num_batch_idx = len(self._internals[dl_idx][num_opt_idx]) - 1
        assert num_batch_idx >= 0
        return self._internals[dl_idx][num_opt_idx][str(num_batch_idx)][-1]
    
    def get_lastest(self, func_name, *args, latest=True, **kwargs):
        results = {}
        if latest:
            for dl_idx in range(self.num_dataloaders):
                dl_idx = str(dl_idx)
                if self._internal_type == self._types[0]:
                    latest_result = self._internals[dl_idx][-1]
                else:
                    latest_result = self.get_latest_from_dict(dl_idx)
                func = getattr(latest_result, func_name)
                results.update(func(*args, add_dataloader_idx=self.add_dataloader_idx, **kwargs))
            return results
        else:
            raise NotImplementedError

    def get_batch_pbar_metrics(self, latest=True, *args, **kwargs):
        return self.get_lastest("get_batch_pbar_metrics", *args, latest=latest, **kwargs)

    def get_batch_log_metrics(self, latest=True, *args, **kwargs):
        return self.get_lastest("get_batch_log_metrics", *args, latest=latest, **kwargs)

    def get_epoch_pbar_metrics(self, *args, **kwargs):
        results = {}
        for dl_idx in range(self.num_dataloaders):
            dl_idx = str(dl_idx)
            opt_metrics = self._internals_reduced[dl_idx]
            if isinstance(opt_metrics, defaultdict):
                for opt_metric in opt_metrics.values():
                    metrics_to_log = opt_metric.get_epoch_pbar_metrics(*args, 
                        add_dataloader_idx=self.add_dataloader_idx, **kwargs)
                    results.update(metrics_to_log)
            else:
                metrics_to_log = opt_metrics.get_epoch_pbar_metrics(*args, 
                        add_dataloader_idx=self.add_dataloader_idx, **kwargs)
                results.update(metrics_to_log)
        return results

    def get_epoch_log_metrics(self, *args, **kwargs):
        results = {}
        for dl_idx in range(self.num_dataloaders):
            dl_idx = str(dl_idx)
            opt_metrics = self._internals_reduced[dl_idx]
            if isinstance(opt_metrics, defaultdict):
                for opt_metric in opt_metrics.values():
                    metrics_to_log = opt_metric.get_epoch_log_metrics(*args, 
                        add_dataloader_idx=self.add_dataloader_idx, **kwargs)
                    results.update(metrics_to_log)
            else:
                metrics_to_log = opt_metrics.get_epoch_log_metrics(*args, 
                        add_dataloader_idx=self.add_dataloader_idx, **kwargs)
                results.update(metrics_to_log)
        return results

    def get_forked_metrics(self, *args, **kwargs):
        results = {}
        for dl_idx in range(self.num_dataloaders):
            dl_idx = str(dl_idx)
            opt_metrics = self._internals_reduced[dl_idx]
            if isinstance(opt_metrics, defaultdict):
                for opt_metric in opt_metrics.values():
                    metrics_to_log = opt_metric.get_forked_metrics(*args, 
                        add_dataloader_idx=self.add_dataloader_idx, **kwargs)
                    results.update(metrics_to_log)
            else:
                metrics_to_log = opt_metrics.get_forked_metrics(*args, 
                        add_dataloader_idx=self.add_dataloader_idx, **kwargs)
                results.update(metrics_to_log)
        return results

    def append(self, result, dataloader_idx=None, extra_info: dict = {}):
        if dataloader_idx is None:
            dataloader_idx = 0

        primary_key = f"{dataloader_idx}" 

        # [dataloader_idx][optimizer_idx][training_step_idx] is a list
        if len(extra_info) > 0:
            self._internal_type = self._types[-1]
            # initialize dictionary
            if primary_key not in self._internals:
                self._internals[primary_key] = defaultdict(lambda : defaultdict(list))
                # need to capture reduction by opt_ix
                self._internals_reduced[primary_key] = defaultdict(dict)

            # extract infos
            opt_idx = str(extra_info["opt_idx"])
            batch_idx = str(extra_info["batch_idx"])

            # add to cache
            self._internals[primary_key][opt_idx][batch_idx].append(result)

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
                    num_batch_idx = len(self._internals[dl_idx][str(num_opt_idx)]) - 1
                    
                    # Make sure we didn't create key 
                    assert num_opt_idx >= 0 and num_batch_idx >= 0

                    for opt_idx in range(num_opt_idx + 1):
                        opt_idx = str(opt_idx)
                        opt_outputs = epoch_metrics[opt_idx]
                    
                        # reduce across time first
                        time_reduced_outputs = []
                        for batch_idx in range(num_batch_idx + 1):
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

                        if len(opt_outputs) > 0:
                            self._internals_reduced[dl_idx][str(opt_idx)] = opt_outputs
                else:
                    # no need to reduce as called only once
                    if len(epoch_metrics) == 1:
                        reduced_epoch_metrics = deepcopy(epoch_metrics[0])
                    else:
                        reduced_epoch_metrics = epoch_metrics[0].__class__.reduce_on_epoch_end(deepcopy(epoch_metrics))

                    self._internals_reduced[dl_idx] = reduced_epoch_metrics
            
            self.has_reduced = True

    def __repr__(self):
        return self._internals.__repr__()

class EpochLoopResult:
    """
    This class is responsible to cache all logging metrics which happened during one epoch 
    """

    def __init__(self, trainer, stage):
        self.trainer = trainer
        self._stage = stage
        self._internals = {}
        self._dataloader_idx = None
        self._split_idx = None
        self._opt_idx = None
        self._has_been_reduced = False

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

    def reset_model(self, model_ref):
        model_ref._results = Result()
        model_ref._current_hook_fx_name = ''
        model_ref._current_fx_name = ''

    def cache_result(self):
        model_ref = self.trainer.get_model()

        # extract hook information
        hook_result = model_ref._results
        fx_name = model_ref._current_hook_fx_name
        if fx_name == '':
            fx_name = model_ref._current_fx_name
        dataloader_idx = model_ref._current_dataloader_idx

        # add only if anything as been logged
        # default len is 1 due to _internals
        if len(hook_result) > 1:

            if fx_name not in self._internals:
                self._internals[fx_name] = HookResults(fx_name)

            extra_info = {}
            if self.has_split_and_opt_idx:
                extra_info = self.extra_info

            self._internals[fx_name].append(deepcopy(hook_result), 
                                                     dataloader_idx=dataloader_idx,
                                                     extra_info=extra_info)

            # update logged_metrics, progress_bar_metrics, callback_metrics
            self.update_logger_connector()
        
        # reset _results, fx_name
        self.reset_model(model_ref)

    def update_logger_connector(self):
        logger_connector = self.trainer.logger_connector

        # update pbar
        batch_pbar_metrics = self.get_latest_batch_pbar_metrics(include_forked_originals=True)
        self.add_progress_bar_metrics(batch_pbar_metrics)

        # update logged_metrics
        batch_log_metrics = self.get_latest_batch_log_metrics()
        logger_connector.logged_metrics.update(batch_log_metrics)

        # update callback_metrics
        logger_connector.callback_metrics.update(logger_connector.progress_bar_metrics)
        logger_connector.callback_metrics.update(logger_connector.logged_metrics)

    def get_latest_batch_log_metrics(self):
        results = {}
        for fx_name, hook_result in self._internals.items():
            results.update(hook_result.get_batch_log_metrics(latest=True,
                include_forked_originals=False))
        return results

    def get_latest_batch_pbar_metrics(self):
        results = {}
        for fx_name, hook_result in self._internals.items():
            results.update(hook_result.get_batch_pbar_metrics(latest=True, 
                include_forked_originals=False))
        return results  
    
    @property
    def has_reduced(self):
        hook_results = self._internals.values()
        return len(hook_results) == sum([h.has_reduced for h in hook_results])

    def _auto_reduce_results_on_epoch_end(self):
        if not self.has_reduced:
            for fx_name, hook_result in self._internals.items():
                hook_result.auto_reduce_results_on_epoch_end()

    def __repr__(self):
        return f"{self.__class__.__name__}(stage={self._stage}, internals={self._internals})"

    def reset(self):
        self = EpochLoopResult(self.trainer, self._stage)

    def get_epoch_pbar_metrics(self):
        if not self.has_reduced:
            self._auto_reduce_results_on_epoch_end()
        epoch_pbar_metrics = {}
        for fx_name, hook_result in self._internals.items():
            epoch_pbar_metrics.update(hook_result.get_epoch_pbar_metrics())
        return epoch_pbar_metrics 

    def get_epoch_log_metrics(self):
        if not self.has_reduced:
            self._auto_reduce_results_on_epoch_end()
        epoch_log_metrics = {}
        for fx_name, hook_result in self._internals.items():
            epoch_log_metrics.update(hook_result.get_epoch_log_metrics() )     
        return epoch_log_metrics

    def get_forked_metrics(self):
        if not self.has_reduced:
            self._auto_reduce_results_on_epoch_end()
        forked_metrics = {}
        for fx_name, hook_result in self._internals.items():
            forked_metrics.update(hook_result.get_forked_metrics())     
        return forked_metrics