import numbers
from copy import copy
from typing import Optional, Dict, Union, Sequence, Callable, MutableMapping, Any

import torch
from torch import Tensor

from pytorch_lightning.metrics.converters import _sync_ddp_if_available


class Result(Dict):

    def __init__(
            self,
            minimize: Optional[Tensor] = None,
            early_stop_on: Optional[Tensor] = None,
            checkpoint_on: Union[Tensor, bool, None] = None,
            hiddens: Optional[Tensor] = None,
    ):

        super().__init__()

        if early_stop_on is not None:
            self.early_stop_on = early_stop_on
        if checkpoint_on is not None and checkpoint_on:
            self.checkpoint_on = checkpoint_on
        if hiddens is not None:
            self.hiddens = hiddens.detach()
        if minimize is not None:
            err = 'Minimize can only be used in training_step, training_step_end, training_epoch_end'
            self._assert_grad_tensor_metric('minimize', minimize, err)
            self.minimize = minimize

        if minimize is not None and checkpoint_on is None:
            self.checkpoint_on = minimize.detach()

        self['meta'] = {
            '_internal': {
                '_reduce_on_epoch': False,
                'batch_sizes': []
            }
        }

    def __getattr__(self, key: str) -> Any:
        try:
            if key == 'callback_metrics':
                return self.get_callback_metrics()
            elif key == 'batch_log_metrics':
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
        # ensure reserve keys are tensors and detached
        if key in {'checkpoint_on', 'early_stop_on'}:
            self._assert_tensor_metric(key, val)
            if val is not None and isinstance(val, torch.Tensor):
                val = val.detach()

        # ensure anything else that is a tensor is detached
        elif isinstance(val, torch.Tensor) and key != 'minimize':
            val = val.detach()

        self[key] = val

    def _assert_tensor_metric(self, name: str, potential_metric: Union[bool, Tensor, None, Any]):
        if potential_metric is not None and not isinstance(potential_metric, bool):
            assert isinstance(potential_metric, Tensor), f'{name} must be a torch.Tensor'

    def _assert_grad_tensor_metric(self, name: str, x: Union[torch.Tensor, Any], additional_err: str = ''):
        if x is not None:
            assert isinstance(x, Tensor), f'{name} must be a torch.Tensor'
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
            sync_dist_group: Optional[Any] = None
    ):
        # no metrics should be logged with graphs
        if not enable_graph and isinstance(value, torch.Tensor):
            value = value.detach()

        # sync across ddp
        if sync_dist and isinstance(value, (torch.Tensor, numbers.Number)):
            value = _sync_ddp_if_available(value, group=sync_dist_group, reduce_op=sync_dist_op)

        if 'meta' not in self:
            self.__setitem__('meta', {})

        # if user requests both step and epoch, then we split the metric in two automatically
        # one will be logged per step. the other per epoch
        if on_step and on_epoch:
            # set step version
            step_name = f'step_{name}'
            self.__set_meta(step_name, value, prog_bar, logger,
                            on_step=True, on_epoch=False,
                            reduce_fx=reduce_fx, tbptt_reduce_fx=tbptt_reduce_fx, tbptt_pad_token=tbptt_pad_token)
            self.__setitem__(step_name, value)

            # set epoch version
            epoch_name = f'epoch_{name}'
            self.__set_meta(epoch_name, value, prog_bar, logger, on_step=False, on_epoch=True,
                            reduce_fx=reduce_fx, tbptt_reduce_fx=tbptt_reduce_fx, tbptt_pad_token=tbptt_pad_token)
            self.__setitem__(epoch_name, value)
        else:
            self.__set_meta(name, value,
                            prog_bar, logger,
                            on_step, on_epoch,
                            reduce_fx,
                            tbptt_reduce_fx=tbptt_reduce_fx, tbptt_pad_token=tbptt_pad_token)

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
            tbptt_reduce_fx: Callable
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
            tbptt_pad_token=tbptt_pad_token
        )

        self['meta'][name] = meta

        # track whether any input requires reduction on epoch end
        _internal = self['meta']['_internal']
        _internal['_reduce_on_epoch'] = max(_internal['_reduce_on_epoch'], on_epoch)

    def track_batch_size(self, batch_size):
        meta = self['meta']
        meta['_internal']['batch_sizes'].append(batch_size)

    def get_batch_sizes(self):
        meta = self['meta']
        return torch.tensor(meta['_internal']['batch_sizes'])

    def get_callback_metrics(self) -> dict:
        result = {
            'early_stop_on': self.early_stop_on,
            'checkpoint_on': self.checkpoint_on
        }

        return result

    def get_batch_log_metrics(self) -> dict:
        """
        Gets the metrics to log at the end of the batch step
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if k == '_internal':
                continue
            if options['logger'] and options['on_step']:
                result[k] = self[k]
        return result

    def get_epoch_log_metrics(self) -> dict:
        """
        Gets the metrics to log at the end of the batch step
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if k == '_internal':
                continue
            if options['logger'] and options['on_epoch']:
                result[k] = self[k]
        return result

    def get_epoch_pbar_metrics(self):
        """
        Gets the metrics to log at the end of the batch step
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if k == '_internal':
                continue
            if options['prog_bar'] and options['on_epoch']:
                result[k] = self[k]
        return result

    def get_batch_pbar_metrics(self):
        """
        Gets the metrics to log at the end of the batch step
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if k == '_internal':
                continue
            if options['prog_bar'] and options['on_step']:
                result[k] = self[k]
        return result

    def detach(self):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self.__setitem__(k, v.detach())

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
            newone[k] = copy(v)
        return newone

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
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                if name not in {'checkpoint_on', 'early_stop_on', 'minimize'}:
                    default_padding_idx = meta[name]['tbptt_pad_token']
                    break

        # pad across each key individually
        for name, value in result.items():
            is_reserved = name in {'checkpoint_on', 'early_stop_on', 'minimize'}
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):

                if is_reserved:
                    padding_key = default_padding_idx
                else:
                    padding_key = meta[name]['tbptt_pad_token']
                padded = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=padding_key)
                result[name] = padded

                # also update the result
                if meta and not is_reserved:
                    meta[name]['value'] = padded
        if meta:
            result['meta'] = meta
        return result

    @classmethod
    def reduce_on_epoch_end(cls, outputs):
        # get the batch sizes for all outputs
        batch_sizes = torch.stack([x.get_batch_sizes() for x in outputs]).view(-1)

        meta = outputs[0]['meta']
        result = cls()
        result = recursive_gather(outputs, result)
        recursive_stack(result)


        for k, option in meta.items():
            if k == '_internal':
                continue

            if option['on_epoch']:
                fx = option['reduce_fx']
                if fx == torch.mean:
                    reduced_val = weighted_mean(result[k], batch_sizes)
                else:
                    reduced_val = fx(result[k])

                result[k] = reduced_val

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
            if k == 'meta':
                continue

            # pick the reduce fx
            if k in ['checkpoint_on', 'early_stop_on', 'minimize']:
                tbptt_reduce_fx = torch.mean
            else:
                tbptt_reduce_fx = meta[k]['tbptt_reduce_fx']
            result[k] = tbptt_reduce_fx(value)

        result['meta'] = meta
        return result

    @property
    def should_reduce_on_epoch_end(self) -> bool:
        return self['meta']['_internal']['_reduce_on_epoch']

    def drop_hiddens(self):
        if 'hiddens' in self:
            del self['hiddens']

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


def recursive_gather(outputs: Sequence[dict], result: Optional[MutableMapping] = None) -> Optional[MutableMapping]:
    for out in outputs:
        if 'meta' in out:
            del out['meta']

        for k, v in out.items():
            if isinstance(v, dict):
                v = recursive_gather([v], result)

            if k not in result:
                result[k] = []

            result[k].append(v)

    return result


def recursive_stack(result: MutableMapping):
    for k, v in result.items():
        if isinstance(v, dict):
            recursive_stack(v)

        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            v = torch.stack(v)
            result[k] = v


def recursive_padded_stack(result: MutableMapping):
    for k, v in result.items():
        if isinstance(v, dict):
            recursive_stack(v)

        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            v = torch.stack(v)
            result[k] = v


class TrainResult(Result):

    def __init__(
            self,
            minimize: Optional[Tensor] = None,
            early_stop_on: Tensor = None,
            checkpoint_on: Union[Tensor, bool] = None,
            hiddens: Optional[Tensor] = None,
    ):
        """
        Used in train loop to auto-log to a logger or progress bar without needing to define
        a train_step_end or train_epoch_end method

        Example::

            def training_step(self, batch, batch_idx):
                loss = ...
                result = pl.TrainResult(loss)
                result.log('train_loss', loss)
                return result

            # without val/test loop can model checkpoint or early stop
            def training_step(self, batch, batch_idx):
                loss = ...
                result = pl.TrainResult(loss, early_stop_on=loss, checkpoint_on=loss)
                result.log('train_loss', loss)
                return result

        Args:
            early_stop_on:
            checkpoint_on:
            hiddens:
        """

        super().__init__(minimize, early_stop_on, checkpoint_on, hiddens)

    def log(
            self,
            name,
            value,
            prog_bar: bool = False,
            logger: bool = True,
            on_step: bool = True,
            on_epoch: bool = False,
            reduce_fx: Callable = torch.mean,
            tbptt_reduce_fx: Callable = torch.mean,
            tbptt_pad_token: int = 0,
            enable_graph: bool = False,
            sync_dist: bool = False,
            sync_dist_op: Union[Any, str] = 'mean',
            sync_dist_group: Optional[Any] = None
    ):
        """
        Log a key, value

        Example::

            result.log('train_loss', loss)

            # defaults used
            result.log(
                name,
                value,
                on_step=True,
                on_epoch=False,
                logger=True,
                prog_bar=False,
                reduce_fx=torch.mean,
                enable_graph=False
            )


        Args:
            name: key name
            value: value name
            prog_bar: if True logs to the progress base
            logger: if True logs to the logger
            on_step: if True logs the output of validation_step or test_step
            on_epoch: if True, logs the output of the training loop aggregated
            reduce_fx: Torch.mean by default
            tbptt_reduce_fx: function to reduce on truncated back prop
            tbptt_pad_token: token to use for padding
            enable_graph: if True, will not auto detach the graph
            sync_dist: if True, reduces the metric across GPUs/TPUs
            sync_dist_op: the op to sync across
            sync_dist_group: the ddp group
        """
        super().log(name=name,
                    value=value,
                    prog_bar=prog_bar,
                    logger=logger,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    reduce_fx=reduce_fx,
                    enable_graph=enable_graph,
                    sync_dist=sync_dist,
                    sync_dist_group=sync_dist_group,
                    sync_dist_op=sync_dist_op,
                    tbptt_pad_token=tbptt_pad_token,
                    tbptt_reduce_fx=tbptt_reduce_fx)

    def log_dict(
            self,
            dictionary: dict,
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
            sync_dist_group: Optional[Any] = None
    ):
        """
        Log a dictonary of values at once

        Example::

            values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
            result.log_dict(values)

        Args:
            dictionary: key value pairs (str, tensors)
            prog_bar: if True logs to the progress base
            logger: if True logs to the logger
            on_step: if True logs the output of validation_step or test_step
            on_epoch: if True, logs the output of the training loop aggregated
            reduce_fx: Torch.mean by default
            tbptt_reduce_fx: function to reduce on truncated back prop
            tbptt_pad_token: token to use for padding
            enable_graph: if True, will not auto detach the graph
            sync_dist: if True, reduces the metric across GPUs/TPUs
            sync_dist_op: the op to sync across
            sync_dist_group: the ddp group:
        """
        for k, v in dictionary.items():
            self.log(name=k,
                     value=v,
                     prog_bar=prog_bar,
                     logger=logger,
                     on_step=on_step,
                     on_epoch=on_epoch,
                     reduce_fx=reduce_fx,
                     enable_graph=enable_graph,
                     sync_dist=sync_dist,
                     sync_dist_group=sync_dist_group,
                     sync_dist_op=sync_dist_op,
                     tbptt_pad_token=tbptt_pad_token,
                     tbptt_reduce_fx=tbptt_reduce_fx)


class EvalResult(Result):

    def __init__(
            self,
            early_stop_on: Optional[Tensor] = None,
            checkpoint_on: Optional[Tensor] = None,
            hiddens: Optional[Tensor] = None,
    ):
        """
        Used in val/train loop to auto-log to a logger or progress bar without needing to define
        a _step_end or _epoch_end method

        Example::

            def validation_step(self, batch, batch_idx):
                loss = ...
                result = EvalResult()
                result.log('val_loss', loss)
                return result

            def test_step(self, batch, batch_idx):
                loss = ...
                result = EvalResult()
                result.log('val_loss', loss)
                return result

        Args:
            early_stop_on:
            checkpoint_on:
            hiddens:
        """

        super().__init__(None, early_stop_on, checkpoint_on, hiddens)

    def log(
            self,
            name,
            value,
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
            sync_dist_group: Optional[Any] = None
    ):
        """
        Log a key, value

        Example::

            result.log('val_loss', loss)

            # defaults used
            result.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=False,
                reduce_fx=torch.mean
            )


        Args:
            name: key name
            value: value name
            prog_bar: if True logs to the progress base
            logger: if True logs to the logger
            on_step: if True logs the output of validation_step or test_step
            on_epoch: if True, logs the output of the training loop aggregated
            reduce_fx: Torch.mean by default
            tbptt_reduce_fx: function to reduce on truncated back prop
            tbptt_pad_token: token to use for padding
            enable_graph: if True, will not auto detach the graph
            sync_dist: if True, reduces the metric across GPUs/TPUs
            sync_dist_op: the op to sync across
            sync_dist_group: the ddp group
        """
        super().log(name=name,
                    value=value,
                    prog_bar=prog_bar,
                    logger=logger,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    reduce_fx=reduce_fx,
                    enable_graph=enable_graph,
                    sync_dist=sync_dist,
                    sync_dist_group=sync_dist_group,
                    sync_dist_op=sync_dist_op,
                    tbptt_pad_token=tbptt_pad_token,
                    tbptt_reduce_fx=tbptt_reduce_fx)

    def log_dict(
            self,
            dictionary: dict,
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
            sync_dist_group: Optional[Any] = None
    ):
        """
        Log a dictonary of values at once

        Example::

            values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
            result.log_dict(values)

        Args:
            dictionary: key value pairs (str, tensors)
            prog_bar: if True logs to the progress base
            logger: if True logs to the logger
            on_step: if True logs the output of validation_step or test_step
            on_epoch: if True, logs the output of the training loop aggregated
            reduce_fx: Torch.mean by default
            tbptt_reduce_fx: function to reduce on truncated back prop
            tbptt_pad_token: token to use for padding
            enable_graph: if True, will not auto detach the graph
            sync_dist: if True, reduces the metric across GPUs/TPUs
            sync_dist_op: the op to sync across
            sync_dist_group: the ddp group
        """
        for k, v in dictionary.items():
            self.log(name=k,
                     value=v,
                     prog_bar=prog_bar,
                     logger=logger,
                     on_step=on_step,
                     on_epoch=on_epoch,
                     reduce_fx=reduce_fx,
                     enable_graph=enable_graph,
                     sync_dist=sync_dist,
                     sync_dist_group=sync_dist_group,
                     sync_dist_op=sync_dist_op,
                     tbptt_pad_token=tbptt_pad_token,
                     tbptt_reduce_fx=tbptt_reduce_fx)

    def get_callback_metrics(self) -> dict:
        result = {
            'val_early_stop_on': self.early_stop_on,
            'val_checkpoint_on': self.checkpoint_on
        }

        return result

    def write(self, name, values, filename='predictions.txt'):

        if isinstance(values, Tensor):
            values = values.detach()

        preds = getattr(self, '_predictions', None)
        if preds is None:
            self._predictions = {filename: {name: values}}
        elif filename not in preds:
            preds[filename] = {name: values}
        elif name not in preds[filename]:
            preds[filename][name] = values
        elif isinstance(values, Tensor):
            preds[filename][name] = torch.cat((preds[filename][name], values))
        elif isinstance(values, list):
            preds[filename][name] = torch.cat((preds[filename][name], values))

    def write_dict(self, predictions_dict, filename='predictions.txt'):
        for k, v in predictions_dict.items():
            self.write(k, v, filename)


def weighted_mean(result, weights):
    weights = weights.to(result.device)
    numerator = torch.dot(result.float(), weights.t().float())
    result = numerator / weights.sum().float()
    return result
