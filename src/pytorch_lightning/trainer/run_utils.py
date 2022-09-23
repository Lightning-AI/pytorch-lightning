# Copyright Lightning AI.
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

from typing import Any, Optional
from weakref import proxy

from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning_lite.utilities.data import _auto_add_worker_init_fn
from lightning_lite.utilities.types import _PATH
from lightning_lite.utilities.warnings import PossibleUserWarning
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.auto_restart import _add_capture_metadata_collate
from pytorch_lightning.utilities.data import has_len_all_ranks
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_training
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn


def restore_modules_and_callbacks(trainer, checkpoint_path: Optional[_PATH] = None) -> None:
    # restore modules after setup
    trainer._checkpoint_connector.resume_start(checkpoint_path)
    trainer._checkpoint_connector._restore_quantization_callbacks()
    trainer._checkpoint_connector.restore_model()
    trainer._checkpoint_connector.restore_datamodule()
    if trainer.state.fn == TrainerFn.FITTING:
        # restore callback states
        trainer._checkpoint_connector.restore_callbacks()


def setup_profiler(trainer) -> None:
    local_rank = trainer.local_rank if trainer.world_size > 1 else None
    trainer.profiler._lightning_module = proxy(trainer.lightning_module)
    trainer.profiler.setup(stage=trainer.state.fn._setup_fn, local_rank=local_rank, log_dir=trainer.log_dir)


def log_hyperparams(trainer) -> None:
    if not trainer.loggers:
        return
    # log hyper-parameters
    hparams_initial = None

    # save exp to get started (this is where the first experiment logs are written)
    datamodule_log_hyperparams = trainer.datamodule._log_hyperparams if trainer.datamodule is not None else False

    if trainer.lightning_module._log_hyperparams and datamodule_log_hyperparams:
        datamodule_hparams = trainer.datamodule.hparams_initial
        lightning_hparams = trainer.lightning_module.hparams_initial
        inconsistent_keys = []
        for key in lightning_hparams.keys() & datamodule_hparams.keys():
            lm_val, dm_val = lightning_hparams[key], datamodule_hparams[key]
            if type(lm_val) != type(dm_val):
                inconsistent_keys.append(key)
            elif isinstance(lm_val, Tensor) and id(lm_val) != id(dm_val):
                inconsistent_keys.append(key)
            elif lm_val != dm_val:
                inconsistent_keys.append(key)
        if inconsistent_keys:
            raise MisconfigurationException(
                f"Error while merging hparams: the keys {inconsistent_keys} are present "
                "in both the LightningModule's and LightningDataModule's hparams "
                "but have different values."
            )
        hparams_initial = {**lightning_hparams, **datamodule_hparams}
    elif trainer.lightning_module._log_hyperparams:
        hparams_initial = trainer.lightning_module.hparams_initial
    elif datamodule_log_hyperparams:
        hparams_initial = trainer.datamodule.hparams_initial

    for logger in trainer.loggers:
        if hparams_initial is not None:
            logger.log_hyperparams(hparams_initial)
        logger.log_graph(trainer.lightning_module)
        logger.save()


"""
Data loading methods
"""


def reset_train_dataloader(trainer: Any, model: Optional["pl.LightningModule"] = None) -> None:
    """Resets the train dataloader and initialises required variables (number of batches, when to validate, etc.).

    Args:
        model: The ``LightningModule`` if calling this outside of the trainer scope.
    """
    source = trainer._data_connector._train_dataloader_source
    pl_module = model or trainer.lightning_module
    has_step = is_overridden("training_step", pl_module)
    enable_training = trainer.limit_train_batches > 0
    if not (source.is_defined() and has_step and enable_training):
        return

    trainer.train_dataloader = trainer._data_connector._request_dataloader(RunningStage.TRAINING)

    if trainer.overfit_batches > 0:
        trainer.train_dataloader = trainer._data_connector._resolve_overfit_batches(
            trainer.train_dataloader, mode=RunningStage.TRAINING
        )

    # automatically add samplers
    trainer.train_dataloader = apply_to_collection(
        trainer.train_dataloader,
        (DataLoader, CombinedLoader),
        trainer._data_connector._prepare_dataloader,
        mode=RunningStage.TRAINING,
    )
    loaders = (
        trainer.train_dataloader.loaders
        if isinstance(trainer.train_dataloader, CombinedLoader)
        else trainer.train_dataloader
    )

    # check the workers recursively
    apply_to_collection(loaders, DataLoader, trainer._data_connector._worker_check, "train_dataloader")

    # add worker_init_fn for correct seeding in worker processes
    apply_to_collection(loaders, DataLoader, _auto_add_worker_init_fn, rank=trainer.global_rank)

    # add collate_fn to collect metadata for fault tolerant training
    if _fault_tolerant_training():
        apply_to_collection(loaders, DataLoader, _add_capture_metadata_collate)

    # wrap the sequence of train loaders to a CombinedLoader object for computing the num_training_batches
    if not isinstance(trainer.train_dataloader, CombinedLoader):
        trainer.train_dataloader = CombinedLoader(loaders, trainer._data_connector.multiple_trainloader_mode)

    module = model or trainer.lightning_module or trainer.datamodule
    orig_train_batches = trainer.num_training_batches = (
        len(trainer.train_dataloader)
        if has_len_all_ranks(trainer.train_dataloader, trainer.strategy, module)
        else float("inf")
    )
    if orig_train_batches == 0:
        return

    # store epoch of dataloader reset for reload_dataloaders_every_n_epochs
    trainer._last_train_dl_reload_epoch = trainer.current_epoch

    if isinstance(trainer.limit_train_batches, int):
        trainer.num_training_batches = min(orig_train_batches, trainer.limit_train_batches)
    elif trainer.num_training_batches != float("inf"):
        trainer.num_training_batches = int(orig_train_batches * trainer.limit_train_batches)
    elif trainer.limit_train_batches != 1.0:
        raise MisconfigurationException(
            "When using an `IterableDataset`, `Trainer(limit_train_batches)` must be `1.0` or an int."
            "An int specifies `num_training_batches` to use."
        )

    if isinstance(trainer.val_check_interval, int):
        trainer.val_check_batch = trainer.val_check_interval
        if trainer.val_check_batch > trainer.num_training_batches and trainer.check_val_every_n_epoch is not None:
            raise ValueError(
                f"`val_check_interval` ({trainer.val_check_interval}) must be less than or equal "
                f"to the number of the training batches ({trainer.num_training_batches}). "
                "If you want to disable validation set `limit_val_batches` to 0.0 instead."
                "If you want to validate based on the total training batches, set `check_val_every_n_epoch=None`."
            )
    else:
        if not has_len_all_ranks(trainer.train_dataloader, trainer.strategy, module):
            if trainer.val_check_interval == 1.0:
                trainer.val_check_batch = float("inf")
            else:
                raise MisconfigurationException(
                    "When using an IterableDataset for `train_dataloader`,"
                    " `Trainer(val_check_interval)` must be `1.0` or an int. An int k specifies"
                    " checking validation every k training batches."
                )
        else:
            trainer.val_check_batch = int(trainer.num_training_batches * trainer.val_check_interval)
            trainer.val_check_batch = max(1, trainer.val_check_batch)

    if trainer.loggers and trainer.num_training_batches < trainer.log_every_n_steps:
        rank_zero_warn(
            f"The number of training batches ({trainer.num_training_batches}) is smaller than the logging interval"
            f" Trainer(log_every_n_steps={trainer.log_every_n_steps}). Set a lower value for log_every_n_steps if"
            " you want to see logs for the training epoch.",
            category=PossibleUserWarning,
        )

    if (
        trainer.num_training_batches == 0
        and trainer.limit_train_batches > 0.0
        and isinstance(trainer.limit_train_batches, float)
        and orig_train_batches != float("inf")
    ):
        min_percentage = 1.0 / orig_train_batches
        raise MisconfigurationException(
            f"You requested to check {trainer.limit_train_batches} of the `train_dataloader` but"
            f" {trainer.limit_train_batches} * {orig_train_batches} < 1. Please increase the"
            f" `limit_train_batches` argument. Try at least"
            f" `limit_train_batches={min_percentage}`"
        )


def reset_val_dataloader(trainer: Any, model: Optional["pl.LightningModule"] = None) -> None:
    """Resets the validation dataloader and determines the number of batches.

    Args:
        model: The ``LightningModule`` if called outside of the trainer scope.
    """
    source = trainer._data_connector._val_dataloader_source
    pl_module = trainer.lightning_module or model
    has_step = is_overridden("validation_step", pl_module)
    enable_validation = trainer.limit_val_batches > 0
    if source.is_defined() and has_step and enable_validation:
        # store epoch of dataloader reset for reload_dataloaders_every_n_epochs
        # it should not reload again if it has already reloaded during sanity_check
        if trainer.state.fn == TrainerFn.FITTING and (
            (trainer.sanity_checking and trainer.fit_loop.epoch_loop._should_check_val_epoch())
            or not trainer.sanity_checking
        ):
            trainer._last_val_dl_reload_epoch = trainer.current_epoch

        trainer.num_val_batches, trainer.val_dataloaders = trainer._data_connector._reset_eval_dataloader(
            RunningStage.VALIDATING, model=pl_module
        )


def reset_test_dataloader(trainer, model: Optional["pl.LightningModule"] = None) -> None:
    """Resets the test dataloader and determines the number of batches.

    Args:
        model: The ``LightningModule`` if called outside of the trainer scope.
    """
    source = trainer._data_connector._test_dataloader_source
    pl_module = trainer.lightning_module or model
    has_step = is_overridden("test_step", pl_module)
    enable_testing = trainer.limit_test_batches > 0
    if source.is_defined() and has_step and enable_testing:
        trainer.num_test_batches, trainer.test_dataloaders = trainer._data_connector._reset_eval_dataloader(
            RunningStage.TESTING, model=pl_module
        )


def reset_predict_dataloader(trainer: Any, model: Optional["pl.LightningModule"] = None) -> None:
    """Resets the predict dataloader and determines the number of batches.

    Args:
        model: The ``LightningModule`` if called outside of the trainer scope.
    """
    source = trainer._data_connector._predict_dataloader_source
    pl_module = trainer.lightning_module or model
    enable_prediction = trainer.limit_predict_batches > 0
    if source.is_defined() and enable_prediction:
        trainer.num_predict_batches, trainer.predict_dataloaders = trainer._data_connector._reset_eval_dataloader(
            RunningStage.PREDICTING, model=pl_module
        )


def reset_train_val_dataloaders(trainer, model: Optional["pl.LightningModule"] = None) -> None:
    """Resets train and val dataloaders if none are attached to the trainer.

    The val dataloader must be initialized before training loop starts, as the training loop
    inspects the val dataloader to determine whether to run the evaluation loop.

    Args:
        model: The ``LightningModule`` if called outside of the trainer scope.

    .. deprecated:: v1.7
        This method is deprecated in v1.7 and will be removed in v1.9.
        Please use ``Trainer.reset_{train,val}_dataloader`` instead.
    """
    rank_zero_deprecation(
        "`Trainer.reset_train_val_dataloaders` has been deprecated in v1.7 and will be removed in v1.9."
        " Use `Trainer.reset_{train,val}_dataloader` instead"
    )
    if trainer.train_dataloader is None:
        reset_train_dataloader(trainer, model=model)
    if trainer.val_dataloaders is None:
        reset_val_dataloader(trainer, model=model)
