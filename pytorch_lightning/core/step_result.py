from typing import Optional, Dict
from torch import Tensor
import torch
from copy import copy


class Result(Dict):

    def __init__(
            self,
            minimize: Optional[Tensor] = None,
            early_stop_on: Tensor = None,
            checkpoint_on: Tensor = None,
            hiddens: Optional[Tensor] = None
    ):

        super().__init__()

        if early_stop_on is not None:
            self.early_stop_on = early_stop_on
        if checkpoint_on is not None:
            self.checkpoint_on = checkpoint_on
        if hiddens is not None:
            self.hiddens = hiddens
        if minimize is not None:
            self.minimize = minimize

        if minimize is not None and early_stop_on is None:
            self.early_stop_on = minimize.detach()
        if minimize is not None and checkpoint_on is None:
            self.checkpoint_on = minimize.detach()

    def __getattr__(self, key):
        try:
            if key == 'callback_metrics':
                return self.get_callback_metrics()
            elif key == 'batch_log_metrics':
                return self.get_batch_log_metrics()
            elif key == 'batch_pbar_metrics':
                return self.get_batch_pbar_metrics()
            else:
                return self[key]
        except KeyError:
            return None

    def __setattr__(self, key, val):
        # ensure reserve keys are tensors and detached
        if key in {'hiddens', 'checkpoint_on', 'early_stop_on'}:
            self._assert_tensor_metric(key, val)
            if val is not None:
                val = val.detach()

        # ensure minimize is a tensor and has grads
        elif key == 'minimize':
            err = 'Minimize can only be used in training_end, training_step_end, training_epoch_end'
            self._assert_grad_tensor_metric(key, val, err)

        # ensure anything else that is a tensor is detached
        elif isinstance(val, torch.Tensor):
            val = val.detach()

        self[key] = val

    def _assert_tensor_metric(self, name, x):
        if x is not None:
            assert isinstance(x, Tensor), f'{name} must be a torch.Tensor'

    def _assert_grad_tensor_metric(self, name, x, additional_err: str = None):
        if x is not None:
            assert isinstance(x, Tensor), f'{name} must be a torch.Tensor'
            m = f'{name} must have a computational graph.'

            if additional_err:
                m += f' {additional_err}'
            assert x.grad_fn is not None, m

    def log(
            self,
            name,
            value,
            prog_bar=False,
            logger=True,
            reduce_on_batch_end=False,
            reduce_on_epoch_end=True,
            reduce_fx=torch.mean
    ):
        if 'meta' not in self:
            self.__setitem__('meta', {})
        self.__set_meta(name, value, prog_bar, logger, reduce_on_batch_end, reduce_on_epoch_end, reduce_fx)

        # set the value
        self.__setitem__(name, value)

    def __set_meta(self, name, value, prog_bar, logger, reduce_on_batch_end, reduce_on_epoch_end, reduce_fx):
        # set the meta for the item
        meta_value = value
        if isinstance(meta_value, torch.Tensor):
            meta_value = meta_value.detach()
        meta = dict(
            prog_bar=prog_bar,
            logger=logger,
            reduce_on_batch_end=reduce_on_batch_end,
            reduce_on_epoch_end=reduce_on_epoch_end,
            reduce_fx=reduce_fx,
            value=meta_value
        )
        self['meta'][name] = meta

    def get_callback_metrics(self):
        result = {
            'early_stop_on': self.early_stop_on,
            'checkpoint_on': self.checkpoint_on
        }

        return result

    def get_batch_log_metrics(self):
        """
        Gets the metrics to log at the end of the batch step
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if options['logger']:
                result[k] = options['value']
        return result

    def get_batch_pbar_metrics(self):
        """
        Gets the metrics to log at the end of the batch step
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if options['prog_bar']:
                result[k] = options['value']
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
        meta = outputs[0]['meta']
        result = Result()
        result = recursive_gather(outputs, result)
        recursive_stack(result)
        result['meta'] = meta
        return result


def recursive_gather(outputs, result=None):
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


def recursive_stack(result):
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
            checkpoint_on: Tensor = None,
            hiddens: Optional[Tensor] = None
    ):

        super().__init__(minimize, early_stop_on, checkpoint_on, hiddens)

    def log(
            self,
            name,
            value,
            prog_bar=False,
            logger=True,
            reduce_on_batch_end=True,
            reduce_on_epoch_end=False,
            reduce_fx=torch.mean
    ):
        super().log(name, value, prog_bar, logger, reduce_on_batch_end, reduce_on_epoch_end, reduce_fx)


class EvalResult(Result):

    def __init__(
            self,
            early_stop_on: Tensor = None,
            checkpoint_on: Tensor = None,
            hiddens: Optional[Tensor] = None
    ):

        super().__init__(None, early_stop_on, checkpoint_on, hiddens)

    def log(
            self,
            name,
            value,
            prog_bar=False,
            logger=True,
            reduce_on_batch_end=False,
            reduce_on_epoch_end=True,
            reduce_fx=torch.mean
    ):
        super().log(name, value, prog_bar, logger, reduce_on_batch_end, reduce_on_epoch_end, reduce_fx)


if __name__ == '__main__':
    import torch
    result = TrainResult()
    result.hiddens = torch.tensor(1)
    result.log('some', 123)
    print(result)
    result.minimize = torch.tensor(1)