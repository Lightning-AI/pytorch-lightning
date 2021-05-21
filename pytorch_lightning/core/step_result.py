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
import numbers
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Tuple, Union

import torch
from torch import Tensor
from torchmetrics import Metric

from pytorch_lightning.utilities.distributed import sync_ddp_if_available, tpu_distributed


@dataclass
class Metadata:
    prog_bar: bool = False
    logger: bool = True
    on_step: bool = False
    on_epoch: bool = True
    reduce_fx: Callable = torch.mean
    tbptt_reduce_fx: Callable = torch.mean
    tbptt_pad_token: int = 0
    dataloader_idx: Optional[int] = None

    @property
    def forked(self) -> bool:
        return self.on_step and self.on_epoch

    def names(self, name: str) -> List[str]:
        names = [name]
        # check both
        if self.on_step:
            names += name + '_step'
        if self.on_epoch:
            names += name + '_epoch'
        return names

@dataclass
class Result:
    data: Any  # TODO: Union[Tensor, Metric]?
    meta: Metadata = field(repr=False)
    batch_sizes: List[int] = field(default_factory=list, init=False)

    @staticmethod
    def extract_batch_size(batch: Any) -> int:
        try:
            return Result._extract_batch_size(batch)
        except RecursionError:
            return 1

    @staticmethod
    def _extract_batch_size(batch: Any) -> int:
        """
        Recursively unpack a batch to find a torch.Tensor.

        Returns:
            ``len(tensor)`` when found, or ``1`` when it hits an empty or non iterable.
        """
        if isinstance(batch, torch.Tensor):
            size = batch.size(0)
        elif isinstance(batch, str):
            return len(batch)
        elif isinstance(batch, dict):
            sample = next(iter(batch.values()), 1)
            size = Result._extract_batch_size(sample)
        elif isinstance(batch, Iterable):
            sample = next(iter(batch), 1)
            size = Result._extract_batch_size(sample)
        else:
            size = 1
        return size


class ResultCollection(dict):

    def __init__(self) -> None:
        super().__init__()
        self.minimize: Optional[Tensor] = None
        self.should_reduce_on_epoch_end = False

    #@staticmethod
    #def removesuffix(s: str, suffix: str) -> str:
    #    # available from Python 3.9
    #    if suffix and s.endswith(suffix):
    #        return s[:-len(suffix)]
    #    return s

    #@staticmethod
    #def _parse_key(key: str) -> str:
    #    key = ResultCollection.removesuffix(key, '_epoch')
    #    key = ResultCollection.removesuffix(key, '_step')
    #    return key

    #def __getitem__(self, key: str) -> Result:
    #    if not isinstance(key, str):
    #        raise ValueError(f'`Result` keys must be `str`, found: {key}')
    #    if key in self:
    #        return super().__getitem__(key)
    #    # try removing `_epoch` and `_step` suffixes
    #    key = self._parse_key(key)
    #    return super().__getitem__(key)

    def get_callback_metrics(self):
        return self.items()

    def get_logger_metrics(self):
        pass
        #ret = {}
        #for item in self.items():
        #    for name in item.names(): # names knows whether it is forked
        #        ret[name] = item.data
        # checks whether is forked and returns all
        # return self.items_prefixes()

    @staticmethod
    def _sync(
        value,
        sync_fn: Optional[Callable] = None,
        sync_dist: bool = False,
        sync_dist_op: Union[Any, str] = 'mean',
        sync_dist_group: Optional[Any] = None,
        device: torch.device = None,
    ):
        """Sync across workers when using distributed training"""
        if not isinstance(value, (torch.Tensor, numbers.Number)):
            return value

        sync_fn = sync_fn or sync_ddp_if_available
        dist_available = torch.distributed.is_available() and torch.distributed.is_initialized() or tpu_distributed()
        if not sync_dist or not dist_available:
            return value

        # TODO: Find a way to make the reduction only once, so we don't need to clone.
        if isinstance(value, torch.Tensor):
            value = value.clone()
        else:
            value = torch.tensor(value, device=device, dtype=torch.float)
        return sync_fn(value, group=sync_dist_group, reduce_op=sync_dist_op)

    @property
    def minimize(self) -> Optional[Tensor]:
        return self.get('minimize', None)

    @minimize.setter
    def minimize(self, val: Optional[torch.Tensor]) -> None:
        if val is not None:
            if not isinstance(val, Tensor):
                raise ValueError(f"`Result.minimize` must be a `torch.Tensor`, found: {val}")
            if val.grad_fn is None:
                raise RuntimeError("`Result.minimize` must have a `grad_fn`")
        self['minimize'] = val

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
        batch_size: Optional[int] = None,
    ):
        """See :meth:`~pytorch_lightning.core.lightning.LightningModule.log`"""
        # no metrics should be logged with graphs
        if not enable_graph and isinstance(value, torch.Tensor):
            value = value.detach()

        # TODO: should this be in the caller?
        value = self._sync(
            value,
            sync_fn=sync_fn,
            sync_dist=sync_dist,
            sync_dist_op=sync_dist_op,
            sync_dist_group=sync_dist_group,
            device=device,
        )

        if isinstance(value, torch.Tensor) and value.device.type == "xla":
            value = value.cpu()

        result = Result(
            value,
            Metadata(
                prog_bar=prog_bar,
                logger=logger,
                on_step=on_step,
                on_epoch=on_epoch,
                reduce_fx=reduce_fx,
                tbptt_reduce_fx=tbptt_reduce_fx,
                tbptt_pad_token=tbptt_pad_token,
                dataloader_idx=dataloader_idx,
            ),
        )
        if batch_size is None:
            batch_size = Result.extract_batch_size(value)
        result.batch_sizes.append(batch_size)
        self[name] = result
        self.should_reduce_on_epoch_end |= on_epoch

    @staticmethod
    def _add_dl_idx(key: str, dl_idx: Union[int, None]) -> str:
        if dl_idx is not None:
            return f"{key}/dataloader_idx_{dl_idx}"
        return key

    @staticmethod
    def _filter(self: 'Result', fields: List[str], add_dataloader_idx: bool = False) -> Dict[str, '_METRIC']:  # TODO
        result = {}
        for k, item in self.items():
            # check if we need to add the suffix
            if 'on_step' in fields and 'on_epoch' not in fields:
                k += '_step'
            elif 'on_step' not in fields and 'on_epoch' in fields:
                k += '_epoch'

            if all(getattr(item.meta, f, False) for f in fields):
                k = Result._add_dl_idx(k, item.meta.dataloader_idx)
                result[k] = item.data
        return result

    def get_batch_log_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, '_METRIC']:
        """Gets the metrics to log at the end of the batch"""
        # TODO: remove dl idx
        results = self._filter(self, ['logger', 'on_step'], add_dataloader_idx=add_dataloader_idx)
        for k, v in results:
            if isinstance(v, Metric) and v._forward_cache is not None:
                results[k] = v._foward_cache.detach()
        return results

    def get_epoch_log_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, '_METRIC']:
        """Gets the metrics to log at the end of epoch"""
        results = self._filter(self, ['logger', 'on_epoch'], add_dataloader_idx=add_dataloader_idx)
        for k, v in results:
            if isinstance(v, Metric) and v._forward_cache is not None:
                results[k] = v._foward_cache.compute().detach()
            # TODO: this?
            # if k in self and not options['on_epoch'] and isinstance(self[k], Metric):
            #     # compute for reuse later
            #     self[k].compute()
        return results

    def get_batch_pbar_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, '_METRIC']:
        """Gets the metrics to include in the progress_bar at the end of epoch"""
        results = self._filter(self, ['prog_bar', 'on_step'], add_dataloader_idx=add_dataloader_idx)
        for k, v in results:
            if isinstance(v, Metric) and v._forward_cache is not None:
                results[k] = v._foward_cache.detach()
        return results

    def get_epoch_pbar_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, '_METRIC']:
        """Gets the metrics to include in the progress_bar at the end of epoch"""
        results = self._filter(self, ['prog_bar', 'on_epoch'], add_dataloader_idx=add_dataloader_idx)
        for k, v in results:
            if isinstance(v, Metric) and v._forward_cache is not None:
                results[k] = v._foward_cache.compute().detach()
            # TODO: this?
            # if k in self and not options['on_epoch'] and isinstance(self[k], Metric):
            #     # compute for reuse later
            #     self[k].compute()
        return results

    def get_forked_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, '_METRIC']:
        results = self._filter(self, [], add_dataloader_idx=add_dataloader_idx)
        for k, v in results:
            if isinstance(v, Metric) and v._forward_cache is not None:
                results[k] = v._foward_cache.compute().detach()
        return results

    def detach(self) -> 'ResultCollection':
        for k, item in self.items():
            if isinstance(item.data, torch.Tensor):
                item.data = item.data.detach()
        return self

    def to(self, *args, **kwargs) -> 'ResultCollection':
        """Move all data to the given device."""
        for k, item in self.items():
            if isinstance(item.data, torch.Tensor):
                item.data = item.data.to(*args, **kwargs)
        return self

    def cpu(self) -> 'Result':
        """Move all data to CPU."""
        return self.to(device="cpu")

    # TODO: need this with detach?
    #def __copy__(self):
    #    newone = type(self)()
    #    for k, v in self.items():
    #        if isinstance(v, torch.Tensor):
    #            v = v.detach()
    #        newone[k] = copy(v)
    #    return newone

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
            if name != 'minimize' and isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                default_padding_idx = meta[name]['tbptt_pad_token']
                break

        # pad across each key individually
        for name, value in result.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
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
    def reduce_across_time(cls, time_outputs: List['Result']) -> 'Result':
        # auto-reduce across time for tbptt
        meta = time_outputs[0].meta

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

    def get_non_metrics_keys(self) -> List[str]:
        """This function is used to filter metric keys for which the value isn't a Metric"""
        return [k for k, v in self.items() if not isinstance(v.data, Metric)]

    def reset(self) -> None:
        """Call at the end of epoch to reset all metric objects"""
        for item in self.values():
            if isinstance(item.data, Metric):
                item.data.reset()


def choose_last(x):
    if isinstance(x, (torch.Tensor, list)):
        return x[-1]
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = x[k][-1]


def recursive_gather(outputs: List[Result], result: Result) -> Result:
    for out in outputs:
        for k, item in out.items():
            if isinstance(item.data, dict):
                in_d = result.get(k, Result())
                result[k] = recursive_gather([item], in_d)
            elif isinstance(item.data, Metric):
                # if v is a metric, just keep one of them,
                # don't keep on adding a list of them
                result[k] = item
            else:
                result.setdefault(k, [])
                result[k].append(item)
    return result


def recursive_stack(result: MutableMapping):
    for k, item in result.items():
        if isinstance(item.data, dict):
            recursive_stack(item.data)

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
