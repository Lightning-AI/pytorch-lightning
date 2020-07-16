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

        self.early_stop_on = early_stop_on
        self.checkpoint_on = checkpoint_on

        self._hiddens = hiddens
        self.minimize = minimize

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f'Missing attribute "{key}"')

    def __setattr__(self, key, val):
        self[key] = val

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

    @property
    def hiddens(self):
        return self._hiddens

    @hiddens.setter
    def hiddens(self, x):
        if x is not None:
            assert isinstance(x, Tensor), 'hiddens must be a torch.Tensor'
            self._hiddens = x
            self.__setitem__('hiddens', x)

    @property
    def checkpoint_on(self):
        # use minimize as default if no checkpoint_on is passed
        if 'checkpoint_on' not in self:
            minimize = self.__getitem__('minimize')
            self.__setitem__('checkpoint_on', minimize)

        return self.__getitem__('checkpoint_on')

    @checkpoint_on.setter
    def checkpoint_on(self, x):
        if x is not None:
            assert isinstance(x, Tensor), 'checkpoint_on must be a torch.Tensor'
            self.__setitem__('checkpoint_on', x.detach())

    @property
    def early_stop_on(self):
        # use minimize as default if no checkpoint_on is passed
        if 'early_stop_on' not in self:
            minimize = self.__getitem__('minimize')
            self.__setitem__('early_stop_on', minimize)

        return self.__getitem__('early_stop_on')

    @early_stop_on.setter
    def early_stop_on(self, x):
        if x is not None:
            assert isinstance(x, Tensor), 'early_stop_on must be a torch.Tensor'
            self.__setitem__('early_stop_on', x.detach())

    @property
    def minimize(self):
        return self.__getitem__('minimize')

    @minimize.setter
    def minimize(self, x):
        if x is not None:
            assert isinstance(x, Tensor), 'metric to minimize must be a torch.Tensor'
            m = 'the metric to minimize must have a computational graph. Minimize ' \
                'can only be used in training_end, training_step_end, training_epoch_end'
            assert x.grad_fn is not None, m
            self.__setitem__('minimize', x)

    @property
    def callback_metrics(self):
        result = {
            'early_stop_on': self.early_stop_on,
            'checkpoint_on': self.checkpoint_on
        }

        return result

    @property
    def batch_log_metrics(self):
        """
        Gets the metrics to log at the end of the batch step
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if options['logger']:
                result[k] = options['value']
        return result

    @property
    def batch_pbar_metrics(self):
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
            if isinstance(v, torch.Tensor) and v.grad_fn is not None:
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
    result = EvalResult()
    result.minimize = 2
    result.log('some', 123)
    print(result)
    result.minimize = torch.tensor(1)