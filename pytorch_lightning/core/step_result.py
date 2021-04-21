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
"""Result class for easier logging and epoch-wise reduction."""

import numbers
from copy import copy
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torchmetrics import Metric

from pytorch_lightning.utilities.distributed import sync_ddp_if_available, tpu_distributed


class Result(Dict):

    def __init__(self, minimize: Optional[Tensor] = None):
        super().__init__()

        if minimize is not None:
            err = 'Minimize can only be used in training_step, training_step_end, training_epoch_end'
            self._assert_grad_tensor_metric('minimize', minimize, err)
            self.minimize = minimize

        self['meta'] = {'_internal': {'_reduce_on_epoch': False, 'batch_sizes': []}}

    def __getitem__(self, key: Union[str, Any]) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            return super().__getitem__(f'{key}_step')

    def __getattr__(self, key: str) -> Any:
        try:
            if key == 'batch_log_metrics':
                return self.get_batch_log_metrics()
            elif key == 'batch_pbar_metrics':
                return self.get_batch_pbar_metrics()
            elif key == 'epoch_log_metrics':
                return self.get_epoch_log_metrics()
            elif key == 'epoch_pbar_metrics':
                return self.get_epoch_pbar_metrics()
            else:
                return self[key]
        except KeyError:
            return None

    def __setattr__(self, key: str, val: Union[Tensor, Any]):
        # ensure tensors are detached
        if isinstance(val, torch.Tensor) and key != 'minimize':
            val = val.detach()
        self[key] = val

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self.update(d)

    def _assert_grad_tensor_metric(self, name: str, x: Union[torch.Tensor, Any], additional_err: str = ''):
        if x is not None:
            if not isinstance(x, Tensor):
                raise TypeError(f'{name} must be a torch.Tensor')

            m = f'{name} must have a computational graph.'

            if additional_err:
                m += f' {additional_err}'
            assert x.grad_fn is not None, m

    def log(
        self,
        name: str,
        value: Any,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: bool = False,
        on_epoch: bool = True,
        reduce_fx: Callable = torch.mean,
        tbptt_reduce_fx: Callable = torch.mean,
        tbptt_pad_token: int = 0,
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_op: Union[Any, str] = 'mean',
        sync_dist_group: Optional[Any] = None,
        sync_fn: Callable = None,
        dataloader_idx: Optional[int] = None,
        device: torch.device = None,
    ):
        # no metrics should be logged with graphs
        if not enable_graph and isinstance(value, torch.Tensor):
            value = value.detach()

        # sync across workers when using distributed training
        sync_fn = sync_fn or sync_ddp_if_available

        if sync_dist and isinstance(value, (torch.Tensor, numbers.Number)):
            is_dist_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
            # TODO: Find a way to make the reduction only once, so we don't need to clone.
            if (is_dist_initialized or tpu_distributed) and isinstance(value, torch.Tensor):
                value = value.clone()
            else:
                value = torch.tensor(value, device=device, dtype=torch.float)
            value = sync_fn(value, group=sync_dist_group, reduce_op=sync_dist_op)

        if isinstance(value, torch.Tensor) and value.device.type == "xla":
            value = value.cpu()

        if 'meta' not in self:
            self.__setitem__('meta', {})

        # if user requests both step and epoch, then we split the metric in two automatically
        # one will be logged per step. the other per epoch
        was_forked = False
        if on_step and on_epoch:
            was_forked = True

            # set step version
            step_name = f'{name}_step'

            self.__set_meta(
                step_name,
                value,
                prog_bar,
                logger,
                on_step=True,
                on_epoch=False,
                reduce_fx=reduce_fx,
                tbptt_reduce_fx=tbptt_reduce_fx,
                tbptt_pad_token=tbptt_pad_token,
                forked=False,
                dataloader_idx=dataloader_idx,
            )

            self.__setitem__(step_name, value)

            # set epoch version
            epoch_name = f'{name}_epoch'

            self.__set_meta(
                epoch_name,
                value,
                prog_bar,
                logger,
                on_step=False,
                on_epoch=True,
                reduce_fx=reduce_fx,
                tbptt_reduce_fx=tbptt_reduce_fx,
                tbptt_pad_token=tbptt_pad_token,
                forked=False,
                dataloader_idx=dataloader_idx,
            )
            self.__setitem__(epoch_name, value)

        # always log the original metric
        self.__set_meta(
            name,
            value,
            prog_bar,
            logger,
            on_step,
            on_epoch,
            reduce_fx,
            tbptt_reduce_fx=tbptt_reduce_fx,
            tbptt_pad_token=tbptt_pad_token,
            forked=was_forked,
            dataloader_idx=dataloader_idx,
        )

        # set the value
        self.__setitem__(name, value)

    def __set_meta(
        self,
        name: str,
        value: Any,
        prog_bar: bool,
        logger: bool,
        on_step: bool,
        on_epoch: bool,
        reduce_fx: Callable,
        tbptt_pad_token: int,
        tbptt_reduce_fx: Callable,
        forked: bool,
        dataloader_idx: Union[int, None],
    ):
        # set the meta for the item
        meta_value = value
        meta = dict(
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,
            value=meta_value,
            tbptt_reduce_fx=tbptt_reduce_fx,
            tbptt_pad_token=tbptt_pad_token,
            forked=forked,
            dataloader_idx=dataloader_idx,
        )

        self['meta'][name] = meta

        # track whether any input requires reduction on epoch end
        _internal = self['meta']['_internal']
        _internal['_reduce_on_epoch'] = max(_internal['_reduce_on_epoch'], on_epoch)

    def track_batch_size(self, batch):
        batch_size = Result.extract_batch_size(batch)
        Result.attach_batch_size(batch_size, self)

    @staticmethod
    def extract_batch_size(batch):
        try:
            batch_size = Result.unpack_batch_size(batch)
        except RecursionError:
            batch_size = 1
        return batch_size

    @staticmethod
    def attach_batch_size(batch_size: Union[int, None], result: 'Result') -> None:
        if batch_size is not None:
            meta = result['meta']
            meta['_internal']['batch_sizes'].append(batch_size)

    def get_batch_sizes(self):
        meta = self['meta']
        return torch.tensor(meta['_internal']['batch_sizes'])

    def _add_dataloader_idx(self, k: str, dataloader_idx: Union[int, None], add_dataloader_idx: bool) -> str:
        if dataloader_idx is not None and add_dataloader_idx:
            return f"{k}/dataloader_idx_{dataloader_idx}"
        return k

    def get_batch_log_metrics(self, include_forked_originals=True, add_dataloader_idx=False) -> dict:
        """
        Gets the metrics to log at the end of the batch step

        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if k == '_internal':
                continue

            if options['forked'] and not include_forked_originals:
                continue

            dl_key = self._add_dataloader_idx(k, options["dataloader_idx"], add_dataloader_idx)

            if options['logger'] and options['on_step']:
                if isinstance(self[k], Metric) and self[k]._forward_cache is not None:
                    result[dl_key] = self[k]._forward_cache.detach()
                else:
                    result[dl_key] = self[k]

        return result

    def get_epoch_log_metrics(self, add_dataloader_idx=False) -> dict:
        """
        Gets the metrics to log at the end of epoch
        """
        result = {}
        meta = self['meta']
        for k, options in meta.items():
            if k == '_internal':
                continue

            if options['forked']:
                continue

            dl_key = self._add_dataloader_idx(k, options["dataloader_idx"], add_dataloader_idx)

            if options['logger'] and options['on_epoch']:
                if isinstance(self[k], Metric):
                    result[dl_key] = self[k].compute().detach()
                else:
                    result[dl_key] = self[k]

            if k in self and not options['on_epoch'] and isinstance(self[k], Metric):
                # compute for reuse later
                self[k].compute()

        return result

    def get_epoch_pbar_metrics(self, add_dataloader_idx=False):
        """
        Gets the metrics to log at the end of epoch
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if k == '_internal':
                continue

            if options['forked']:
                continue

            dl_key = self._add_dataloader_idx(k, options["dataloader_idx"], add_dataloader_idx)

            if options['prog_bar'] and options['on_epoch']:
                if isinstance(self[k], Metric):
                    result[dl_key] = self[k].compute().detach()
                else:
                    result[dl_key] = self[k]

            if k in self and not options['on_epoch'] and isinstance(self[k], Metric):
                # compute for reuse later
                self[k].compute()

        return result

    def get_forked_metrics(self, add_dataloader_idx=False):
        """
        Gets the metrics to log at the end of epoch
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if k == '_internal':
                continue

            dl_key = self._add_dataloader_idx(k, options["dataloader_idx"], add_dataloader_idx)

            if options['forked']:
                if isinstance(self[k], Metric):
                    result[dl_key] = self[k].compute().detach()
                else:
                    result[dl_key] = self[k]

        return result

    def get_batch_pbar_metrics(self, include_forked_originals=True, add_dataloader_idx=False):
        """
        Gets the metrics to log at the end of the batch step
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if k == '_internal':
                continue

            if options['forked'] and not include_forked_originals:
                continue

            dl_key = self._add_dataloader_idx(k, options["dataloader_idx"], add_dataloader_idx)

            if options['prog_bar'] and options['on_step']:
                if isinstance(self[k], Metric) and self[k]._forward_cache is not None:
                    result[dl_key] = self[k]._forward_cache
                else:
                    result[dl_key] = self[k]

        return result

    def detach(self) -> 'Result':
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self.__setitem__(k, v.detach())
        return self

    def to(self, *args, **kwargs) -> 'Result':
        """Move all self attributes to the given device."""
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self.__setitem__(k, v.to(*args, **kwargs))
        return self

    def cpu(self) -> 'Result':
        """Move all self attributes to CPU."""
        return self.to(torch.device("cpu"))

    def __repr__(self):
        self_copy = self.copy()

        if 'meta' in self_copy:
            del self_copy['meta']

        return str(self_copy)

    def __str__(self):
        copy = self.copy()
        del copy['meta']

        return str(copy)

    def __copy__(self):
        newone = type(self)()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            newone[k] = copy(v)
        return newone

    @staticmethod
    def unpack_batch_size(sample):
        """
        Recursively unpack sample to find a torch.Tensor.
        returns len(tensor) when found, or 1 when it hits an empty or non iterable.
        """
        if isinstance(sample, torch.Tensor):
            size = sample.size(0)
        elif isinstance(sample, str):
            return len(sample)
        elif isinstance(sample, dict):
            sample = next(iter(sample.values()), 1)
            size = Result.unpack_batch_size(sample)
        elif isinstance(sample, Iterable):
            sample = next(iter(sample), 1)
            size = Result.unpack_batch_size(sample)
        else:
            size = 1
        return size

    @classmethod
    def gather(cls, outputs):
        meta = outputs[0].get('meta')
        result = cls()
        result = recursive_gather(outputs, result)
        recursive_stack(result)

        if meta:
            result['meta'] = meta
        return result

    @classmethod
    def padded_gather(cls, outputs):
        meta = outputs[0].get('meta')
        result = cls()
        result = recursive_gather(outputs, result)

        # find the padding used for other values
        default_padding_idx = 0
        for name, value in result.items():
            if (
                name != 'minimize' and isinstance(value, list) and len(value) > 0
                and isinstance(value[0], torch.Tensor)
            ):
                default_padding_idx = meta[name]['tbptt_pad_token']
                break

        # pad across each key individually
        for name, value in result.items():
            if (isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor)):
                padding_key = default_padding_idx if name == 'minimize' else meta[name]['tbptt_pad_token']
                padded = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=padding_key)
                result[name] = padded

                # also update the result
                if meta and name != "minimize":
                    meta[name]['value'] = padded
        if meta:
            result['meta'] = meta
        return result

    @classmethod
    def reduce_on_epoch_end(cls, outputs):
        # get the batch sizes for all outputs
        batch_sizes = []
        meta = {}
        for x in outputs:
            batch_sizes.append(x.get_batch_sizes())
            meta.update(x['meta'])

        batch_sizes = torch.stack(batch_sizes).view(-1)

        result = cls()
        result = recursive_gather(outputs, result)
        recursive_stack(result)

        for k, option in meta.items():
            if k == '_internal' or isinstance(result[k], Metric):
                continue

            # for forked metrics don't reduce, just take the last val
            if option['forked']:
                result[k] = choose_last(result[k])
                continue

            if option['on_epoch']:
                fx = option['reduce_fx']
                if fx == torch.mean:
                    if isinstance(result[k], list):
                        result[k] = torch.tensor(result[k]).float()
                    try:
                        reduced_val = weighted_mean(result[k], batch_sizes)
                    # todo: specify the expected Exceptions to come
                    except Exception:
                        reduced_val = torch.mean(result[k])
                else:
                    reduced_val = fx(result[k])

                result[k] = reduced_val
            else:
                del result[k]

        result['meta'] = meta
        return result

    @classmethod
    def reduce_across_time(cls, time_outputs):
        # auto-reduce across time for tbptt
        meta = time_outputs[0]['meta']

        result = cls()
        result = recursive_gather(time_outputs, result)
        recursive_stack(result)

        for k, value in result.items():
            if k in ['meta', 'extra'] or isinstance(value, Metric):
                continue

            # pick the reduce fx
            tbptt_reduce_fx = torch.mean if k == "minimize" else meta[k]['tbptt_reduce_fx']

            if isinstance(value, list):
                value = torch.tensor(value)

            if isinstance(value, dict):
                # TODO: recursive reduce:
                _recursive_fx_apply(value, tbptt_reduce_fx)
            else:
                result[k] = tbptt_reduce_fx(value.float())

        result['meta'] = meta
        return result

    def dp_reduce(self):
        for k, value in self.items():
            if k == 'meta' or isinstance(value, Metric):
                continue

            if isinstance(value, list):
                value = torch.tensor(value)

            self[k] = value.mean(dim=-1)

    @property
    def should_reduce_on_epoch_end(self) -> bool:
        return self['meta']['_internal']['_reduce_on_epoch']

    def rename_keys(self, map_dict: dict):
        """
        Maps key values to the target values. Useful when renaming variables in mass.

        Args:
            map_dict:
        """
        meta = self.meta
        for source, dest in map_dict.items():
            # map the main keys
            self[dest] = self[source]
            del self[source]

            # map meta
            meta[dest] = meta[source]
            del meta[source]

    def reset(self) -> None:
        """
        Call at the end of epoch to reset all metric objects
        """
        for k, value in self.items():
            if isinstance(value, Metric):
                value.reset()


def choose_last(x):
    if isinstance(x, (torch.Tensor, list)):
        return x[-1]
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = x[k][-1]


def recursive_gather(outputs: Sequence[dict], result: Optional[MutableMapping] = None) -> Optional[MutableMapping]:
    for out in outputs:
        if 'meta' in out:
            del out['meta']

        for k, v in out.items():
            # support manual opt where the user does not return a minimize key
            if k == 'minimize' and v is None:
                continue

            if isinstance(v, dict):
                in_d = result.get(k, {})
                v = recursive_gather([v], in_d)
                result[k] = v
            else:
                if isinstance(v, Metric):
                    # if v is a metric, just keep one of them,
                    # don't keep on adding a list of them
                    result[k] = v
                else:
                    if k not in result:
                        result[k] = []
                    result[k].append(v)

    return result


def recursive_stack(result: MutableMapping):
    for k, v in result.items():
        if isinstance(v, dict):
            recursive_stack(v)

        result[k] = collate_tensors(v)


def _recursive_fx_apply(input: dict, fx):
    for k, v in input.items():
        if isinstance(v, list):
            v = torch.tensor(v)

        if isinstance(v, torch.Tensor):
            v = fx(v.float())
            input[k] = v
        else:
            _recursive_fx_apply(v, fx)


def collate_tensors(items: Union[List, Tuple]) -> Union[Tensor, List, Tuple]:
    if not items or not isinstance(items, (list, tuple)) or any(not isinstance(item, Tensor) for item in items):
        # items is not a sequence, empty, or contains non-tensors
        return items

    if all(item.ndim == 0 for item in items):
        # all tensors are scalars, we need to stack
        return torch.stack(items)

    if all(item.ndim >= 1 and item.shape[1:] == items[0].shape[1:] for item in items):
        # we can concatenate along the first dimension
        return torch.cat(items)

    return items


def weighted_mean(result, weights):

    if isinstance(result, dict):
        _process_dataloader_aggregated_steps(result, weights)
    else:
        if isinstance(result, list):
            result = torch.tensor(result)

        weights = weights.to(result.device)[:result.size(0)]
        numerator = torch.dot(result.float(), weights.transpose(-1, 0).float())
        result = numerator / weights.sum().float()
    return result


def _process_dataloader_aggregated_steps(result, weights):
    internal_keys = {'meta'}

    moved = False

    for k, v in result.items():
        if k in internal_keys:
            continue

        # make sure v is a tensor
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)

        # move to memory only once
        if not moved:
            weights = weights.to(v.device)
            moved = True

        # move weights to same device as value to reduce
        weights_t = weights[:v.size(0)]

        # weighted mean
        numerator = torch.dot(v.float(), weights_t.transpose(-1, 0).float())
        v = numerator / weights.sum().float()
        result[k] = v
