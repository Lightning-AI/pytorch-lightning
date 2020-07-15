from typing import Optional, Dict
from torch import Tensor
import torch


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

        self.hiddens = hiddens
        self.minimize = minimize

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

    def __repr__(self):
        copy = self.copy()
        del copy['meta']

        return str(copy)

    def __str__(self):
        copy = self.copy()
        del copy['meta']

        return str(copy)


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
    result.log('some', 123)
    print(result)
    result.minimize = torch.tensor(1)