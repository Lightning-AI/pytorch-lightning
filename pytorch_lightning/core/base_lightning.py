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

import copy
import inspect
import logging
import numbers
import types
from abc import ABC
from argparse import Namespace
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric

from pytorch_lightning.core.hooks import CheckpointHooks, DataHooks, ModelHooks
from pytorch_lightning.core.saving import ALLOWED_CONFIG_TYPES, PRIMITIVE_TYPES
from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import AttributeDict, collect_init_args, save_hyperparameters
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import _METRIC_COLLECTION, EPOCH_OUTPUT, STEP_OUTPUT
from pytorch_lightning.utilities.warnings import WarningCache

if TYPE_CHECKING:
    from pytorch_lightning.trainer.connectors.logger_connector.result import Result

warning_cache = WarningCache()
log = logging.getLogger(__name__)


class RootLightningModule(ABC, ModelHooks, DataHooks, CheckpointHooks):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.loaded_optimizer_states_dict = {}

        #: Pointer to the trainer object
        self.trainer = None

        self._distrib_type = None
        self._device_type = None

        #: True if using amp
        self.use_amp: bool = False

        #: The precision used
        self.precision: int = 32

        # optionally can be set by user
        self._example_input_array = None
        self._datamodule = None
        self._results: Optional['Result'] = None
        self._current_fx_name: Optional[str] = None
        self._running_manual_backward: bool = False
        self._current_dataloader_idx: Optional[int] = None
        self._automatic_optimization: bool = True
        self._truncated_bptt_steps: int = 0
        self._param_requires_grad_state = dict()

    def optimizers(self, use_pl_optimizer: bool = True) -> Any:
        raise NotImplementedError

    def lr_schedulers(self) -> Optional[Any]:
        raise NotImplementedError

    @property
    def example_input_array(self) -> Any:
        return self._example_input_array

    @property
    def current_epoch(self) -> int:
        """The current epoch"""
        return self.trainer.current_epoch if self.trainer else 0

    @property
    def global_step(self) -> int:
        """Total training batches seen across all epochs"""
        return self.trainer.global_step if self.trainer else 0

    @property
    def global_rank(self) -> int:
        """ The index of the current process across all nodes and devices. """
        return self.trainer.global_rank if self.trainer else 0

    @property
    def local_rank(self) -> int:
        """ The index of the current process within a single node. """
        return self.trainer.local_rank if self.trainer else 0

    @example_input_array.setter
    def example_input_array(self, example: Any) -> None:
        self._example_input_array = example

    @property
    def datamodule(self) -> Any:
        rank_zero_deprecation(
            "The `LightningModule.datamodule` property is deprecated in v1.3 and will be removed in v1.5."
            " Access the datamodule through using `self.trainer.datamodule` instead."
        )
        return self._datamodule

    @datamodule.setter
    def datamodule(self, datamodule: Any) -> None:
        self._datamodule = datamodule

    @property
    def on_gpu(self):
        """
        True if your model is currently running on GPUs.
        Useful to set flags around the LightningModule for different CPU vs GPU behavior.
        """
        return self.device.type == "cuda"

    @property
    def automatic_optimization(self) -> bool:
        """
        If False you are responsible for calling .backward, .step, zero_grad.
        """
        return self._automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, automatic_optimization: bool) -> None:
        self._automatic_optimization = automatic_optimization

    @property
    def truncated_bptt_steps(self) -> int:
        """
        truncated_bptt_steps: Truncated back prop breaks performs backprop every k steps of much a longer sequence.
        If this is > 0, the training step is passed ``hiddens``.
        """
        return self._truncated_bptt_steps

    @truncated_bptt_steps.setter
    def truncated_bptt_steps(self, truncated_bptt_steps: int) -> None:
        self._truncated_bptt_steps = truncated_bptt_steps

    @property
    def logger(self):
        """ Reference to the logger object in the Trainer. """
        return self.trainer.logger if self.trainer else None

    def _apply_batch_transfer_handler(
        self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: Optional[int] = None
    ) -> Any:
        device = device or self.device
        batch = self.on_before_batch_transfer(batch, dataloader_idx)

        if is_param_in_hook_signature(self.transfer_batch_to_device, 'dataloader_idx'):
            batch = self.transfer_batch_to_device(batch, device, dataloader_idx)
        else:
            warning_cache.warn(
                "`transfer_batch_to_device` hook signature has changed in v1.4."
                " `dataloader_idx` parameter has been added to it. Support for"
                " the old signature will be removed in v1.6", DeprecationWarning
            )
            batch = self.transfer_batch_to_device(batch, device)

        batch = self.on_after_batch_transfer(batch, dataloader_idx)
        return batch

    def print(self, *args, **kwargs) -> None:
        r"""
        Prints only from process 0. Use this in any distributed mode to log only once.

        Args:
            *args: The thing to print. The same as for Python's built-in print function.
            **kwargs: The same as for Python's built-in print function.

        Example::

            def forward(self, x):
                self.print(x, 'in forward')

        """
        if self.trainer.is_global_zero:
            progress_bar = self.trainer.progress_bar_callback
            if progress_bar is not None and progress_bar.is_enabled:
                progress_bar.print(*args, **kwargs)
            else:
                print(*args, **kwargs)

    def log(
        self,
        name: str,
        value: _METRIC_COLLECTION,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Callable = torch.mean,
        tbptt_reduce_fx: Optional = None,  # noqa: Remove in 1.6
        tbptt_pad_token: Optional = None,  # noqa: Remove in 1.6
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_op: Union[Any, str] = 'mean',
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
    ) -> None:
        """
        Log a key, value

        Example::

            self.log('train_loss', loss)

        The default behavior per hook is as follows

        .. csv-table:: ``*`` also applies to the test loop
           :header: "LightningModule Hook", "on_step", "on_epoch", "prog_bar", "logger"
           :widths: 20, 10, 10, 10, 10

           "training_step", "T", "F", "F", "T"
           "training_step_end", "T", "F", "F", "T"
           "training_epoch_end", "F", "T", "F", "T"
           "validation_step*", "F", "T", "F", "T"
           "validation_step_end*", "F", "T", "F", "T"
           "validation_epoch_end*", "F", "T", "F", "T"

        Args:
            name: key to log
            value: value to log
            prog_bar: if True logs to the progress bar
            logger: if True logs to the logger
            on_step: if True logs at this step. None auto-logs at the training_step but not validation/test_step
            on_epoch: if True logs epoch accumulated metrics. None auto-logs at the val/test step but not training_step
            reduce_fx: reduction function over step values for end of epoch. :meth:`torch.mean` by default.
            enable_graph: if True, will not auto detach the graph
            sync_dist: if True, reduces the metric across GPUs/TPUs
            sync_dist_op: the op to sync across GPUs/TPUs
            sync_dist_group: the ddp group to sync across
            add_dataloader_idx: if True, appends the index of the current dataloader to
                the name (when using multiple). If False, user needs to give unique names for
                each dataloader to not mix values
        """
        if tbptt_reduce_fx is not None:
            rank_zero_deprecation(
                '`self.log(tbptt_reduce_fx=...)` is no longer supported. The flag will be removed in v1.6.'
                ' Please, open a discussion explaining your use-case in'
                ' `https://github.com/PyTorchLightning/pytorch-lightning/discussions`'
            )
        if tbptt_pad_token is not None:
            rank_zero_deprecation(
                '`self.log(tbptt_pad_token=...)` is no longer supported. The flag will be removed in v1.6.'
                ' Please, open a discussion explaining your use-case in'
                ' `https://github.com/PyTorchLightning/pytorch-lightning/discussions`'
            )

        # check for invalid values
        apply_to_collection(value, dict, self.__check_not_nested, name)
        apply_to_collection(
            value, object, self._check_allowed, name, value, wrong_dtype=(numbers.Number, Metric, Tensor, dict)
        )

        # set the default depending on the fx_name
        on_step = self.__auto_choose_log_on_step(on_step)
        on_epoch = self.__auto_choose_log_on_epoch(on_epoch)

        assert self._current_fx_name is not None
        self.trainer.logger_connector.check_logging(self._current_fx_name, on_step=on_step, on_epoch=on_epoch)

        # make sure user doesn't introduce logic for multi-dataloaders
        if "/dataloader_idx_" in name:
            raise MisconfigurationException(
                f"You called `self.log` with the key `{name}`"
                " but it should not contain information about `dataloader_idx`"
            )

        sync_fn = partial(
            self._sync,
            sync_fn=self.trainer.training_type_plugin.reduce,
            sync_dist=sync_dist,
            sync_dist_op=sync_dist_op,
            sync_dist_group=sync_dist_group,
            device=self.device,
        )
        value = apply_to_collection(value, (torch.Tensor, numbers.Number), sync_fn)

        assert self._results is not None
        self._results.log(
            name,
            value,
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,
            enable_graph=enable_graph,
            dataloader_idx=(self._current_dataloader_idx if add_dataloader_idx else None),
        )

    def log_dict(
        self,
        dictionary: Mapping[str, _METRIC_COLLECTION],
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Callable = torch.mean,
        tbptt_reduce_fx: Optional = None,  # noqa: Remove in 1.6
        tbptt_pad_token: Optional = None,  # noqa: Remove in 1.6
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_op: Union[Any, str] = 'mean',
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
    ) -> None:
        """
        Log a dictionary of values at once

        Example::

            values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
            self.log_dict(values)

        Args:
            dictionary: key value pairs (str, tensors)
            prog_bar: if True logs to the progress base
            logger: if True logs to the logger
            on_step: if True logs at this step. None auto-logs for training_step but not validation/test_step
            on_epoch: if True logs epoch accumulated metrics. None auto-logs for val/test step but not training_step
            reduce_fx: reduction function over step values for end of epoch. :meth:`torch.mean` by default.
            enable_graph: if True, will not auto detach the graph
            sync_dist: if True, reduces the metric across GPUs/TPUs
            sync_dist_op: the op to sync across GPUs/TPUs
            sync_dist_group: the ddp group sync across
            add_dataloader_idx: if True, appends the index of the current dataloader to
                the name (when using multiple). If False, user needs to give unique names for
                each dataloader to not mix values
        """
        for k, v in dictionary.items():
            self.log(
                name=k,
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
                tbptt_reduce_fx=tbptt_reduce_fx,
                add_dataloader_idx=add_dataloader_idx
            )

    @staticmethod
    def __check_not_nested(value: dict, name: str) -> None:
        # self-imposed restriction. for simplicity
        if any(isinstance(v, dict) for v in value.values()):
            raise ValueError(f'`self.log({name}, {value})` was called, but nested dictionaries cannot be logged')
        return value

    @staticmethod
    def _check_allowed(v: Any, name: str, value: Any) -> None:
        raise ValueError(f'`self.log({name}, {value})` was called, but `{type(v).__name__}` values cannot be logged')

    @staticmethod
    def _sync(*args, **kwargs) -> Any:
        raise NotImplementedError

    def __auto_choose_log_on_step(self, on_step: Optional[bool]) -> bool:
        if on_step is None:
            on_step = False
            on_step |= self._current_fx_name in ('training_step', 'training_step_end')
        return on_step

    def __auto_choose_log_on_epoch(self, on_epoch: Optional[bool]) -> bool:
        if on_epoch is None:
            on_epoch = True
            on_epoch &= self._current_fx_name not in ('training_step', 'training_step_end')
        return on_epoch

    def all_gather(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> Any:
        r"""
        Same as :meth:`torch.nn.Module.forward()`.

        Args:
            *args: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.

        Return:
            Your model's output
        """
        return super().forward(*args, **kwargs)

    def training_step(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def training_step_end(self, *args, **kwargs) -> STEP_OUTPUT:
        raise NotImplementedError

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        raise NotImplementedError

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        raise NotImplementedError

    def validation_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        raise NotImplementedError

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        raise NotImplementedError

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        raise NotImplementedError

    def test_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        raise NotImplementedError

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        raise NotImplementedError

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        raise NotImplementedError

    def configure_callbacks(self) -> List:
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def manual_backward(self, loss: Tensor, optimizer: Optional[Optimizer] = None, *args, **kwargs) -> None:
        raise NotImplementedError

    def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        raise NotImplementedError

    def toggle_optimizer(self, optimizer: Optimizer, optimizer_idx: int):
        raise NotImplementedError

    def untoggle_optimizer(self, optimizer_idx: int):
        raise NotImplementedError

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        raise NotImplementedError

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
        raise NotImplementedError

    def tbptt_split_batch(self, batch: Tensor, split_size: int) -> list:
        raise NotImplementedError

    def summarize(self, mode: Optional[str]) -> Optional[Any]:
        raise NotImplementedError

    def freeze(self) -> None:
        raise NotImplementedError

    def unfreeze(self) -> None:
        raise NotImplementedError

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        r"""
        Implement this to override the default items displayed in the progress bar.
        By default it includes the average loss value, split index of BPTT (if used)
        and the version of the experiment when using a logger.

        .. code-block::

            Epoch 1:   4%|▎         | 40/1095 [00:03<01:37, 10.84it/s, loss=4.501, v_num=10]

        Here is an example how to override the defaults:

        .. code-block:: python

            def get_progress_bar_dict(self):
                # don't show the version number
                items = super().get_progress_bar_dict()
                items.pop("v_num", None)
                return items

        Return:
            Dictionary with the items to be displayed in the progress bar.
        """
        # call .item() only once but store elements without graphs
        running_train_loss = self.trainer.train_loop.running_loss.mean()
        avg_training_loss = None
        if running_train_loss is not None:
            avg_training_loss = running_train_loss.cpu().item()
        elif self.automatic_optimization:
            avg_training_loss = float('NaN')

        tqdm_dict = {}
        if avg_training_loss is not None:
            tqdm_dict["loss"] = f"{avg_training_loss:.3g}"

        module_tbptt_enabled = self.truncated_bptt_steps > 0
        trainer_tbptt_enabled = self.trainer.truncated_bptt_steps is not None and self.trainer.truncated_bptt_steps > 0
        if module_tbptt_enabled or trainer_tbptt_enabled:
            tqdm_dict["split_idx"] = self.trainer.train_loop.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            version = self.trainer.logger.version
            # show last 4 places of long version strings
            version = version[-4:] if isinstance(version, str) else version
            tqdm_dict["v_num"] = version

        return tqdm_dict

    def _verify_is_manual_optimization(self, fn_name):
        if self.automatic_optimization:
            raise MisconfigurationException(
                f'to use {fn_name}, please disable automatic optimization:'
                ' set model property `automatic_optimization` as False'
            )

    @classmethod
    def _auto_collect_arguments(cls, frame=None) -> Tuple[Dict, Dict]:
        """
        Collect all module arguments in the current constructor and all child constructors.
        The child constructors are all the ``__init__`` methods that reach the current class through
        (chained) ``super().__init__()`` calls.

        Args:
            frame: instance frame

        Returns:
            self_arguments: arguments dictionary of the first instance
            parents_arguments: arguments dictionary of the parent's instances
        """
        if not frame:
            frame = inspect.currentframe()

        frame_args = collect_init_args(frame.f_back, [])
        self_arguments = frame_args[-1]

        # set hyper_parameters in child
        self_arguments = self_arguments
        parents_arguments = {}

        # add all arguments from parents
        for args in frame_args[:-1]:
            parents_arguments.update(args)
        return self_arguments, parents_arguments

    def save_hyperparameters(
        self,
        *args,
        ignore: Optional[Union[Sequence[str], str]] = None,
        frame: Optional[types.FrameType] = None
    ) -> None:
        """Save model arguments to ``hparams`` attribute.

        Args:
            args: single object of `dict`, `NameSpace` or `OmegaConf`
                or string names or arguments from class ``__init__``
            ignore: an argument name or a list of argument names from
                class ``__init__`` to be ignored
            frame: a frame object. Default is None

        Example::
            >>> class ManuallyArgsModel(RootLightningModule):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # manually assign arguments
            ...         self.save_hyperparameters('arg1', 'arg3')
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = ManuallyArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg3": 3.14

            >>> class AutomaticArgsModel(RootLightningModule):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # equivalent automatic
            ...         self.save_hyperparameters()
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = AutomaticArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg2": abc
            "arg3": 3.14

            >>> class SingleArgModel(RootLightningModule):
            ...     def __init__(self, params):
            ...         super().__init__()
            ...         # manually assign single argument
            ...         self.save_hyperparameters(params)
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = SingleArgModel(Namespace(p1=1, p2='abc', p3=3.14))
            >>> model.hparams
            "p1": 1
            "p2": abc
            "p3": 3.14

            >>> class ManuallyArgsModel(RootLightningModule):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # pass argument(s) to ignore as a string or in a list
            ...         self.save_hyperparameters(ignore='arg2')
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = ManuallyArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg3": 3.14
        """
        # the frame needs to be created in this file.
        if not frame:
            frame = inspect.currentframe().f_back
        save_hyperparameters(self, *args, ignore=ignore, frame=frame)

    def _set_hparams(self, hp: Union[dict, Namespace, str]) -> None:
        if isinstance(hp, Namespace):
            hp = vars(hp)
        if isinstance(hp, dict):
            hp = AttributeDict(hp)
        elif isinstance(hp, PRIMITIVE_TYPES):
            raise ValueError(f"Primitives {PRIMITIVE_TYPES} are not allowed.")
        elif not isinstance(hp, ALLOWED_CONFIG_TYPES):
            raise ValueError(f"Unsupported config type of {type(hp)}.")

        if isinstance(hp, dict) and isinstance(self.hparams, dict):
            self.hparams.update(hp)
        else:
            self._hparams = hp

    @property
    def hparams(self) -> Union[AttributeDict, dict, Namespace]:
        if not hasattr(self, "_hparams"):
            self._hparams = AttributeDict()
        return self._hparams

    @property
    def hparams_initial(self) -> AttributeDict:
        if not hasattr(self, "_hparams_initial"):
            return AttributeDict()
        # prevent any change
        return copy.deepcopy(self._hparams_initial)

    @property
    def model_size(self) -> float:
        raise NotImplementedError
