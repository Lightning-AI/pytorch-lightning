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
r"""
Finetuning Scheduler Supporters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Classes composed to support scheduled finetuning

"""
import functools
import itertools
import logging
import os
import pathlib
import re
from abc import ABC
from collections import Counter
from collections.abc import KeysView
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml
from torch.nn import Module
from torch.optim.optimizer import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import LightningEnum, rank_zero_info, rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException

log = logging.getLogger(__name__)


@dataclass
class FTSState:
    """Dataclass to encapsulate the
    :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` internal state."""

    _resume_fit_from_ckpt: bool = False
    _ft_epoch: int = 0
    _ft_global_steps: int = 0
    _curr_depth: int = 0
    _best_ckpt_depth: int = 0
    _ft_sync_props: Tuple[Tuple] = (("current_epoch", "_ft_epoch"), ("global_step", "_ft_global_steps"))
    _ft_sync_objects: Tuple = None
    _curr_thawed_params: List = field(default_factory=list)
    _fts_ckpt_metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self._fts_ckpt_metadata = {
            "current_ckpt_depth": self._curr_depth,
            "best_ckpt_depth": self._best_ckpt_depth,
            "best_ckpt_pgs": {},
        }


class FTSCheckpoint(ModelCheckpoint):
    r"""
    Extends/specializes :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` to facilitate multi-phase
    scheduled finetuning. Overrides the
    :meth:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint.on_save_checkpoint` and
    :meth:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint.on_load_checkpoint` hooks to maintain
    additional state (:attr:`current_ckpt_depth`, :attr:`best_ckpt_depth`). Usage of
    :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts_supporters.FTSCheckpoint` is identical to
    :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` and
    :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts_supporters.FTSCheckpoint` will automatically be used
    if a :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` callback is detected.

    .. warning:: :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts_supporters.FTSCheckpoint` is in beta and
        subject to change. The finetuning schedule (FTS) checkpoint functionality is currently experimental and may be
        added directly into :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` in the future if the
        community deems appropriate. For detailed usage information, see
        :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`.
    """

    def __init__(self, *args, **kwargs):
        """
        Attributes:
            current_ckpt_depth (int):
                Used to track the depth of most recently saved checkpoint
            best_ckpt_depth (int):
                Used to track the depth of the best known checkpoint (it may be from a different training depth)
        """
        super().__init__(*args, **kwargs)
        self.current_ckpt_depth = 0
        self.best_ckpt_depth = 0

    def on_init_start(self, trainer: "pl.Trainer") -> None:
        """When the trainer initialization begins, verify a valid callback configuration is present.

        Args:
            trainer: The :class:`~pytorch_lightning.trainer.trainer.Trainer` object

        Raises:
            MisconfigurationException:
                If a :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` callback is not
                found on initialization
                (:attr:`~pytorch_lightning.trainer.trainer.Trainer.finetuning_scheduler_callback` is ``None``)
            MisconfigurationException:
                If :paramref:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler.restore_best` is
                ``True`` and :paramref:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint.save_top_k` is
                either ``None`` or ``0``
            MisconfigurationException:
                If :paramref:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler.restore_best` is
                ``True`` and ``monitor`` is ``None``
        """
        if not trainer.finetuning_scheduler_callback:
            raise MisconfigurationException(
                f"{self.__class__.__name__} is intended for use with a "
                "finetuning scheduler callback such as "
                "pytorch_lightning.callbacks.finetuning_scheduler.FinetuningScheduler."
                "If not using a finetuning scheduler callback, please use the standard ModelCheckpoint callback."
            )
        # note if only saving best ckpt rather than top k > 1, current_ckpt_depth == best_ckpt_depth
        if trainer.finetuning_scheduler_callback.restore_best:
            if not self.save_top_k or self.save_top_k == 0:
                raise MisconfigurationException(
                    f"{type(trainer.finetuning_scheduler_callback)} was directed to restore checkpoints"
                    f"(restore_best=True) but {self.__class__.__name__} is configured to save no intermediate"
                    "checkpoints (save_top_k is 0 or None). Please set save_top_k to a non-zero value or set"
                    "restore_best=False"
                )
            elif not self.monitor:
                raise MisconfigurationException(
                    f"{type(trainer.finetuning_scheduler_callback)} was directed to restore checkpoints"
                    f"(restore_best=True) but {self.__class__.__name__} but has no quantity to monitor (monitor=None)."
                    "Please provide a value to monitor or set restore_best=False."
                )

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Overrides :meth:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint.on_save_checkpoint` to
        maintain multi-phase training depth state.

        Args:
            trainer: the current :class:`~pytorch_lightning.trainer.trainer.Trainer` instance.
            pl_module: the current :class:`~pytorch_lightning.core.lightning.LightningModule` instance.
            checkpoint: the checkpoint dictionary that will be saved.

        Returns:
            Dict[str, Any]: the checkpoint dictionary that will be saved.
        """
        self.current_ckpt_depth = trainer.finetuning_scheduler_callback.curr_depth
        # note, if current score is precisely the best score but a previous depth had the same score the
        # best ckpt depth will be set to the latest (deepest) depth with that score.
        # a future enhancement of per-depth best score mapping could allow more fine-grained control of this behavior
        if self.current_score == self.best_model_score:
            self.best_ckpt_depth = self.current_ckpt_depth
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
            "dirpath": self.dirpath,
            "current_ckpt_depth": self.current_ckpt_depth,
            "best_ckpt_depth": self.best_ckpt_depth,
        }

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]
    ):
        """Overrides :meth:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint.on_load_checkpoint` to
        load multi-phase training depth state.

        Args:
            trainer: the current :class:`~pytorch_lightning.trainer.trainer.Trainer` instance.
            pl_module: the current :class:`~pytorch_lightning.core.lightning.LightningModule` instance.
            callback_state: the callback state returned by
                :meth:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint.on_save_checkpoint`.
        """
        self.best_model_score = callback_state["best_model_score"]
        self.best_model_path = callback_state["best_model_path"]
        self.current_ckpt_depth = callback_state["current_ckpt_depth"]
        self.best_ckpt_depth = callback_state["best_ckpt_depth"]
        # if we're starting a new level from another checkpoint depth, wait_count could be > 0 contingent on the
        # min_delta
        if trainer.finetuning_scheduler_callback.curr_depth > self.best_ckpt_depth:
            if not trainer.finetuning_scheduler_callback.epoch_transitions_only:
                trainer.early_stopping_callback.wait_count = 0
        if trainer.finetuning_scheduler_callback._fts_state._resume_fit_from_ckpt:
            if trainer.finetuning_scheduler_callback.new_incarnation_mode:
                # reset state for new training incarnation at resumption depth
                self.best_ckpt_depth = self.current_ckpt_depth
                self.best_model_path = ""
                self.best_model_score = None
                self.best_k_models = {}
                self.kth_best_model_path = ""
                self.kth_value = None
            else:
                self.best_k_models[self.best_model_path] = self.best_model_score
                _op = max if self.mode == "min" else min
                self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
                self.kth_value = self.best_k_models[self.kth_best_model_path]


class SchedulingMixin(ABC):
    """Functionality for generating, parsing and executing finetuning schedules."""

    # this is just a summary of variables used in this abstract class, the proper initialisation should be done in the
    # child class
    pl_module: "pl.LightningModule"
    ft_schedule: Union[str, Dict]
    max_depth: int
    curr_depth: int
    _fts_state: FTSState

    def init_fts(self) -> None:
        """Initializes the finetuning schedule and prepares the first scheduled level
        1. Generate the default finetuning schedule and/or load it into
        :paramref:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler.ft_schedule`.
        2. Prepare the first scheduled finetuning level, unfreezing the relevant parameters."""
        self.init_ft_sched()
        _, self._fts_state._curr_thawed_params = self.exec_ft_phase(
            self.pl_module, thaw_pl=self.ft_schedule[0]["params"], init_thaw=True
        )

    def gen_or_load_sched(self) -> None:
        """Load an explicitly specified finetuning schedule if one provided, otherwise generate a default one."""
        if not self.ft_schedule and self.max_depth == -1:
            rank_zero_info("No finetuning schedule provided, max_depth set to -1 so iteratively thawing entire model")
        if self.ft_schedule:  # thaw according to an explicit schedule
            self.ft_schedule = (
                self.load_yaml_schedule(self.ft_schedule)
                if not isinstance(self.ft_schedule, Dict)
                else self.ft_schedule
            )
        else:
            self.gen_implicit_schedule(self.pl_module.trainer.log_dir)
            self.ft_schedule = self.pl_module.trainer.training_type_plugin.broadcast(self.ft_schedule)

    def validate_ft_sched(self) -> Tuple[int, int]:
        """Ensure the explicitly specified finetuning schedule has a valid configuration.

        Returns:
            Tuple[int, int]: A tuple of ints specifying:
                1. The depth of the final scheduled phase
                2. The maximum epoch watermark explicitly specified in the schedule
        """
        max_epoch_wm = -1
        max_phase = 0
        named_params = dict(self.pl_module.named_parameters()).keys()
        for depth in self.ft_schedule.keys():
            max_phase = max(max_phase, depth)
            self.parse_phase(depth, named_params)
            if depth > 0:
                curr_max_epoch = self.ft_schedule[depth]["max_transition_epoch"]
                if 0 <= curr_max_epoch <= max_epoch_wm:
                    es_addendum = " depending upon EarlyStopping criteria."
                    rank_zero_info(
                        f"Specified max_transition_epoch of depth {depth}"
                        f"({self.ft_schedule[depth]['max_transition_epoch']}) is less than or equal to a "
                        f"previous max_transition_epoch ({max_epoch_wm}), depth may execute only a single "
                        f"epoch{'.' if self.epoch_transitions_only else es_addendum}"
                    )
                max_epoch_wm = max(max_epoch_wm, curr_max_epoch)
        self.validate_phases_disjoint()
        if self.epoch_transitions_only:
            self.validate_epoch_transitions()
        return max_phase, max_epoch_wm

    def init_ft_sched(self) -> None:
        """Generate the default finetuning schedule and/or load it into
        :paramref:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler.ft_schedule`. Broadcast the
        schedule to ensure it is available for use in a distributed context."""
        self.gen_or_load_sched()
        if self.max_depth == -1:
            self.max_depth = len(self.ft_schedule) - 1
        else:
            self.max_depth = min(self.max_depth, len(self.ft_schedule) - 1)
        max_phase, max_epoch_wm = self.validate_ft_sched()
        # if the final phase is not using EarlyStopping, apply the maximum phase-specified epoch to global max_epochs
        if self.ft_schedule[max_phase]["max_transition_epoch"] >= 0:
            rank_zero_warn(
                f"Final phase max_transition_epoch ({self.ft_schedule[max_phase]['max_transition_epoch']}) "
                f"will be overidden by the greater of max_epochs ({self.pl_module.trainer.max_epochs}) and "
                f"the maximum phase-specified epoch ({max_epoch_wm})."
            )
            self.pl_module.trainer.fit_loop.max_epochs = max(max_epoch_wm, self.pl_module.trainer.max_epochs)

    def validate_epoch_transitions(self) -> None:
        """If not composing :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` and epoch-driven
        stopping criteria (the default behavior) but instead specifying exclusively epoch-driven transitions (:para
        mref:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler.epoch_transitions_only` is
        ``True``), ensure the specified schedule specifies transitions for every phase.

        Raises:
            MisconfigurationException: If the specified schedule does not include epoch-driven transitions for all
                phases.
        """
        missing_transitions = [d for d in self.ft_schedule.keys() if self.ft_schedule[d]["max_transition_epoch"] < 0]
        if missing_transitions:
            raise MisconfigurationException(
                f"epoch_transitions_only specified but some phases "
                f"({', '.join(str(d) for d in missing_transitions)}) are missing a "
                "max_transition_epoch. Please unset epoch_transitions_only or "
                "specify a max_transition_epoch for each phase."
            )

    @rank_zero_only
    def gen_implicit_schedule(self, sched_dir: str) -> None:
        """Generate the default schedule, save it to ``sched_dir`` and load it into
        :attr:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler.ft_schedule`

        Args:
            sched_dir: directory to which the generated schedule should be written. By default will be
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.log_dir`.
        """
        default_ft_schedule = self.gen_ft_schedule(self.pl_module, sched_dir)
        rank_zero_info(f"Generated default finetuning schedule '{default_ft_schedule}' for iterative finetuning")
        self.ft_schedule = self.load_yaml_schedule(default_ft_schedule)

    @staticmethod
    def gen_ft_schedule(module: Module, dump_loc: str) -> os.PathLike:
        """Generate the default finetuning schedule using a naive, 2-parameters per-level heuristic.

        Args:
            module (:class:`~torch.nn.Module`): The :class:`~torch.nn.Module` for which a finetuning schedule will be
                generated
            dump_loc: The directory to which the generated schedule (.yaml) should be written
        Returns:
            os.PathLike: The path to the generated schedule, by default
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.log_dir` with the name
            (:paramref:`~pytorch_lightning.trainer.trainer.lightning_module`.__class__.__name__)_ft_schedule.yaml
        """
        # Note: This initial default finetuning schedule generation approach is intentionally simple/naive but is
        # effective for a suprising fraction of models. Future versions of this callback may use module introspection to
        # generate default schedules that better accommodate more complex structures and specific architectures if the
        # callback proves sufficiently useful.
        log.info(f"Proceeding with dumping default finetuning schedule for {module.__class__.__name__}")
        param_lists = []
        cur_group = []
        model_params = list(module.named_parameters())[::-1]
        for i, (n, _) in enumerate(model_params):
            if i % 2 == 0:
                cur_group = []
                cur_group.append(n)
            else:
                cur_group.append(n)
                param_lists.append(cur_group)
        if len(model_params) % 2 == 1:
            param_lists.append([model_params[-1][0]])
        layer_config = {}
        dump_path = pathlib.Path(dump_loc)
        dump_path.mkdir(exist_ok=True, parents=True)
        ft_schedule_yaml = dump_path / f"{module.__class__.__name__}_ft_schedule.yaml"
        fs = get_filesystem(ft_schedule_yaml)
        layer_config = {}
        for i, l in enumerate(param_lists):
            layer_config[i] = {"params": l}
        with fs.open(ft_schedule_yaml, "w", newline="") as fp:
            yaml.dump(layer_config, fp)
        assert os.access(ft_schedule_yaml, os.F_OK)
        rank_zero_info(f"Finetuning schedule dumped to {ft_schedule_yaml}.")
        return ft_schedule_yaml

    @staticmethod
    def load_yaml_schedule(schedule_yaml_file: str) -> Dict:
        """Load a schedule defined in a .yaml file and transform it into a dictionary.

        Args:
            schedule_yaml_file (str): The .yaml finetuning schedule file

        Raises:
            MisconfigurationException: If the specified schedule file is not found

        Returns:
            Dict: the Dict representation of the finetuning schedule
        """
        try:
            with open(schedule_yaml_file) as df:
                schedule_dict = yaml.load(df, Loader=yaml.FullLoader)
        except FileNotFoundError as fnf:
            error_msg = (
                f"Could not find specified finetuning scheduling file '{schedule_yaml_file}': {fnf}."
                f"Please reconfigure and try again."
            )
            rank_zero_warn(error_msg)
            raise MisconfigurationException(error_msg)
        return schedule_dict

    def parse_phase(self, depth: int, named_params: KeysView) -> None:
        """Expand any regex expressions specified in an ft_schedule phase to fully qualified parameter names.

        Args:
            depth (int): Schedule depth/phase to parse
            named_params (KeysView): The named parameters of the model

        Raises:
            MisconfigurationException: If a specified parameter or regex does not resolve to at least one parameter.
        """
        self.ft_schedule[depth].setdefault("max_transition_epoch", -1)
        orig_params = self.ft_schedule[depth].get("params", [])
        resolved_params = []
        for p in orig_params:
            regex_params = []
            explicit_params = False
            if p in named_params:
                explicit_params = True
                resolved_params.append(p)
            else:
                ppat = re.compile(p)
                regex_params = [n for n in named_params if ppat.match(n)]
                resolved_params.extend(regex_params)
            if not (regex_params or explicit_params):
                raise MisconfigurationException(
                    f"The parameter or regex '{p}' specified in phase {depth} of the "
                    "provided explicit schedule did not match any named parameter in the "
                    "model."
                )
        self.ft_schedule[depth]["params"] = resolved_params

    def validate_phases_disjoint(self) -> None:
        """Validate that the defined schedule does not specify any parameter in multiple phases.

        Raises:
            MisconfigurationException: Provides a list of the parameters specified in more than one phase.
        """
        phase_lists = [self.ft_schedule[d]["params"] for d in self.ft_schedule.keys()]
        params = Counter(list(itertools.chain(*phase_lists)))
        unique_params = Counter(list(set().union(*phase_lists)))
        params.subtract(unique_params)
        dup_params = list(params.elements())
        if dup_params:
            raise MisconfigurationException(
                f"Phases are not disjoint. The following parameters are specified in "
                f"multiple phases: {', '.join(dup_params)}"
            )

    def thaw_to_depth(self, depth: int = None) -> None:
        """Thaw/unfreeze the current
        :paramref:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler.pl_module` to the specified
        finetuning depth (aka level)

        Args:
            depth: The depth/level to which the
                :paramref:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler.pl_module` will be
                thawed. If no depth is is specified,
                :paramref:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler.curr_depth` will be
                used. Defaults to ``None``.
        """
        # configure optimizer parameter groups for next fts level, adding parameter groups beyond
        # restored optimizer state up to current depth
        next_tl = []
        depth = depth or self.curr_depth
        for i, next_tl in self.ft_schedule.items():
            if i <= depth:
                _, self._fts_state._curr_thawed_params = self.exec_ft_phase(self.pl_module, thaw_pl=next_tl["params"])

    @staticmethod
    def add_optimizer_groups(
        module: Module,
        optimizer: Optimizer,
        thawed_pl: List,
        no_decay: Optional[list] = None,
        lr: Optional[float] = None,
        initial_denom_lr: float = 10.0,
    ):
        """Add optimizer parameter groups associated with the next scheduled finetuning depth/level and extend the
        relevent :paramref:`~pytorch_lighting.trainer.trainer.Trainer.lr_schedulers`.

        Args:
            module (:class:`~torch.nn.Module`): The :class:`~torch.nn.Module` from which the target optimizer parameters
                will be read.
            optimizer (:class:`~torch.optim.Optimizer`): The :class:`~torch.optim.Optimizer` to which parameter groups
                will be configured and added.
            thawed_pl: The list of thawed/unfrozen parameters that should be added to the new parameter group(s)
            no_decay: A list of parameters that should always have weight_decay set to 0. e.g.:
                ["bias", "LayerNorm.weight"]. Defaults to ``None``.
            lr: The initial learning rate for the new parameter group(s). If not specified,
                the ``lr`` of the first scheduled finetuning depth will be used. Defaults to ``None``.
            initial_denom_lr: The scaling factor by which to scale the initial learning rate for new
                parameter groups when no initial learning rate is specified. Defaults to 10.0.
        """
        if len(thawed_pl) == 0:
            rank_zero_warn("No thawed parameters passed so no new optimizer groups will be added.")
        else:
            params_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
            denom_lr = initial_denom_lr if lr is None else 1.0
            lr_factor = params_lr / denom_lr
            added_pgs = 0
            if no_decay:
                optimizer.add_param_group(
                    {
                        "params": [
                            p
                            for n, p in module.named_parameters()
                            if not any(nd in n for nd in no_decay) and n in thawed_pl and p.requires_grad
                        ],
                        "lr": lr_factor,
                    }
                )
                optimizer.add_param_group(
                    {
                        "params": [
                            p
                            for n, p in module.named_parameters()
                            if any(nd in n for nd in no_decay) and n in thawed_pl and p.requires_grad
                        ],
                        "weight_decay": 0.0,
                        "lr": lr_factor,
                    }
                )
                added_pgs = 2
            else:
                optimizer.add_param_group(
                    {
                        "params": [p for n, p in module.named_parameters() if n in thawed_pl and p.requires_grad],
                        "lr": lr_factor,
                    }
                )
                added_pgs = 1
            # extend base_lrs for added groups rather than re-initialize lr_scheduler
            for lr_sched in module.trainer.lr_schedulers:
                lr_sched["scheduler"].base_lrs.extend([lr_factor] * added_pgs)

    @staticmethod
    def sync(objs: Tuple, asets: Tuple[Tuple], agg_func: Callable = max):
        """Synchronize sets of object attributes using a given aggregation function.

        Args:
            objs: The target objects to synchronize
            asets: The attribute sets to synchronize
            agg_func: The aggregation function use to synchronize the target object attribute sets. Defaults to max.
        """
        for attrs in asets:
            agg = functools.reduce(agg_func, [getattr(o, a) for o, a in zip(objs, attrs)])
            for o, a in zip(objs, attrs):
                setattr(o, a, agg)

    @staticmethod
    def exec_ft_phase(module: Module, thaw_pl: List, init_thaw: bool = False) -> Tuple[List, List]:
        """Thaw/unfreeze the provided list of parameters in the provided :class:`~torch.nn.Module`

        Args:
            module (:class:`~torch.nn.Module`): The :class:`~torch.nn.Module` that will have parameters selectively
                unfrozen/thawed.
            thaw_pl: The list of parameters that should be thawed/unfrozen in the :class:`~torch.nn.Module`
            init_thaw: If ``True``, modifies message to user accordingly. Defaults to ``False``.

        Returns:
            Tuple[List, List]: A Tuple of two lists.
                1. The list of newly thawed/unfrozen parameters thawed by this function
                2. A list of all currently thawed/unfrozen parameters in the target :class:`~torch.nn.Module`
        """
        thawed_p_names = []
        curr_thawed = []
        for n, p in module.named_parameters():
            if not p.requires_grad and n in thaw_pl:
                p.requires_grad = True
                thawed_p_names.append(n)
            elif p.requires_grad:
                curr_thawed.append(n)
        if thawed_p_names:
            rank_zero_info(
                f"{'Initializing with' if init_thaw else 'Thawed'} the following module parameters: "
                f"{[n for n in thawed_p_names]}"
            )
        curr_thawed.extend(thawed_p_names)
        rank_zero_info(f"The following module parameters are currently thawed: {[n for n in curr_thawed]}")
        return thawed_p_names, curr_thawed


class CallbackDepMixin(ABC):
    """Functionality for validating/managing callback dependencies."""

    class CheckpointCapabilityType(LightningEnum):
        """Define checkpoint callback capabilities.

        >>> # you can match the type with string
        >>> CallbackDepMixin.CheckpointCapabilityType.FTS == 'fts'
        True
        """

        FTS = "fts"
        BASE = "base"

    def _ckpt_capability(self, c: Callback) -> Optional[CheckpointCapabilityType]:
        """Ascertain current checkpoint callback capabilities.

        Args:
            c: The :class:`~pytorch_lighting.callbacks.Callback` to inspect

        Returns:
            Optional[CheckpointCapabilityType]: The ``CheckpointCapabilityType`` of the inspected class

        .. note::
            This may be changed from a nominal subtype approach to a protocol/structural subtype design once Python >=
                3.8 is required
        """
        if isinstance(c, FTSCheckpoint):
            return self.CheckpointCapabilityType.FTS
        elif isinstance(c, ModelCheckpoint) and not isinstance(c, FTSCheckpoint):
            return self.CheckpointCapabilityType.BASE
        else:
            return None

    def _inspect_callback_deps(self, trainer: "pl.Trainer") -> Tuple[bool]:
        """Inspect the trainer :paramref:`~pytorch_lighting.trainer.trainer.Trainer.callbacks` for earlystopping
        and scheduled finetuning capabilities.

        Args:
            trainer (pl.Trainer):  The :class:`~pytorch_lightning.trainer.trainer.Trainer` object to inspect the
                callbacks of

        Returns:
            Tuple[bool]: The ascertained :paramref:`~pytorch_lighting.trainer.trainer.Trainer.callbacks` capabilities
        """
        has_earlystopping, has_fts_ckpt, has_other_ckpt = False, False, False
        for c in trainer.callbacks:
            if isinstance(c, EarlyStopping):
                has_earlystopping = True
            elif self._ckpt_capability(c) == self.CheckpointCapabilityType.FTS:
                has_fts_ckpt = True
            elif self._ckpt_capability(c) == self.CheckpointCapabilityType.BASE:
                has_other_ckpt = True
        return has_earlystopping, has_fts_ckpt, has_other_ckpt

    def _configure_callback_deps(self, trainer: "pl.Trainer") -> List[Callback]:
        """Ensures FTSCheckpoint and :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` callbacks
        are present and configured, removing any
        :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`s if present.

        Args:
            trainer: The :class:`~pytorch_lightning.trainer.trainer.Trainer` object that may have its callbacks
                list altered.

        Returns:
            List[Callback]: A new callback list that includes at least one FTSCheckpoint and EarlyStopping class,
                ensuring the FTSCheckpoint is at the end of list.
        """
        has_earlystopping, has_fts_ckpt, has_other_ckpt = self._inspect_callback_deps(trainer)
        if not any([has_earlystopping, self.epoch_transitions_only, self.gen_ft_sched_only]):
            rank_zero_warn(
                f"{self.__class__.__name__} currently depends upon an EarlyStopping callback. Adding an"
                "EarlyStopping callback with default configuration"
            )
            trainer.callbacks.append(EarlyStopping(monitor="val_loss"))
        if has_earlystopping and self.epoch_transitions_only:
            rank_zero_warn(
                "You have specified an EarlyStopping callback along with epoch_transitions_only. Pruning the "
                "extraneous EarlyStopping callback"
            )
            trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c, EarlyStopping)]
        if not has_fts_ckpt:
            if has_other_ckpt:
                rank_zero_warn(
                    f"{self.__class__.__name__} currently depends upon a finetuning schedule "
                    "capable ModelCheckpoint callback such as FTSCheckpoint. Substituting current "
                    "ModelCheckpoint for FTSCheckpoint"
                )
                # filter out non-fts capable ModelCheckpoint callbacks
                trainer.callbacks = [
                    c for c in trainer.callbacks if not self._ckpt_capability(c) == self.CheckpointCapabilityType.BASE
                ]
            trainer.callbacks.append(FTSCheckpoint(monitor="val_loss", verbose=True))
        # ensure existing callback_connector logic is adhered to. Adding an FTS configuration method to
        # CallbackConnector or forcing users to manually add default EarlyStopping and FTSCheckpoint classes
        # would avoid this callback_connector call
        return trainer._callback_connector._reorder_callbacks(trainer.callbacks)
