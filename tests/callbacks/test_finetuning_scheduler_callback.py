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
import os
import re
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import yaml
from torch import nn

from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, FinetuningScheduler
from pytorch_lightning.callbacks.finetuning_scheduler import FTSCheckpoint
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.callbacks.test_finetuning_callback import ConvBlock, ConvBlockParam
from tests.helpers import BoringModel
from tests.helpers.advanced_models import ParityModuleRNN
from tests.helpers.runif import RunIf


class FinetuningSchedulerBoringModel(BoringModel):
    """Extend :class:`~tests.helpers.BoringModel` to facilitate testing of
    :class:`~pytorch_lightning.callbacks.finetuning_scheduler.FinetuningScheduler` by ensuring deterministic divergence
    and accommodating no_decay list configuration"""

    def __init__(self, diverge_on_epoch: int = 3, no_decay: Optional[List] = None, weight_decay: float = 1.0e-06):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32), nn.Linear(32, 32), nn.Linear(32, 2))
        self.diverge_on_epoch = diverge_on_epoch
        self.no_decay = no_decay
        self.weight_decay = weight_decay

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.val_loss(batch, output)
        self.log("val_loss", loss, prog_bar=False)
        return {"x": loss}

    def val_loss(self, batch, prediction):
        # Make arbitrary val_loss the inverse of train_loss so val_loss diverges when desired
        val_func = (
            torch.zeros_like(prediction) if self.current_epoch >= self.diverge_on_epoch else torch.ones_like(prediction)
        )
        return torch.nn.functional.mse_loss(prediction, val_func)

    def configure_optimizers(self):
        parameters = filter(lambda x: x.requires_grad, self.parameters())
        optimizer = torch.optim.SGD(parameters, lr=1e-3, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [lr_scheduler]


class TestFinetuningScheduler(FinetuningScheduler):
    """Extends :class:`~pytorch_lightning.callbacks.finetuning_scheduler.FinetuningScheduler` to facilitate intra-
    fit state inspection during testing of scheduled finetuning."""

    def __init__(self, expected_state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_state = expected_state
        self.best_ckpt_test_weight = None
        self.restored_best_cnt = 0

    def on_save_checkpoint(
        self, trainer: "Trainer", pl_module: "LightningModule", checkpoint: Dict[str, Any]
    ) -> Dict[int, List[Dict[str, Any]]]:
        self.best_ckpt_test_weight = self.pl_module._modules["layer"]._modules["3"].bias.data.detach().clone()
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def restore_best_ckpt(self) -> None:
        super().restore_best_ckpt()
        assert torch.equal(self.pl_module._modules["layer"]._modules["3"].bias.data, self.best_ckpt_test_weight)
        self.restored_best_cnt += 1

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        state_key = trainer.current_epoch
        current_state = (
            self.curr_depth,
            self.depth_remaining,
            self._fts_state._ft_epoch,
            self._fts_state._fts_ckpt_metadata["current_ckpt_depth"],
            self._fts_state._fts_ckpt_metadata["best_ckpt_depth"],
            len(self._fts_state._fts_ckpt_metadata["best_ckpt_pgs"]),
            len(self._fts_state._curr_thawed_params),
            len(self._internal_optimizer_metadata[0]),
            trainer.checkpoint_callback.current_ckpt_depth,
            trainer.checkpoint_callback.best_ckpt_depth,
        )
        assert current_state == self.expected_state[state_key]
        if self.restore_best:
            assert self.restored_best_cnt == self.curr_depth
        else:
            assert self.restored_best_cnt == 0


@pytest.fixture(scope="function")
def ckpt_set(tmpdir_factory):
    """A fixture that generates a 'best' and 'kth' checkpoint to be used in scheduled finetuning resumption
    testing."""
    seed_everything(42)
    callbacks = [
        FinetuningScheduler(max_depth=1),
        EarlyStopping(monitor="val_loss", patience=1, min_delta=0.001),
        FTSCheckpoint(monitor="val_loss", verbose=True, save_top_k=3),
    ]
    model = FinetuningSchedulerBoringModel()
    trainer = Trainer(default_root_dir=tmpdir_factory.getbasetemp(), callbacks=callbacks)
    trainer.fit(model)
    return {"best": trainer.checkpoint_callback.best_model_path, "kth": trainer.checkpoint_callback.kth_best_model_path}


@pytest.fixture(scope="function")
def boring_ft_schedule(tmpdir_factory) -> Tuple[Path, Dict]:
    """Generates a default finetuning schedule for 'implicit' testing, a modified one for 'explicit' mode and an
    epoch-driven transitions only one for epoch_transitions_only testing."""
    seed_everything(42)
    callbacks = [FinetuningScheduler(gen_ft_sched_only=True)]
    model = FinetuningSchedulerBoringModel()
    tmpdir = tmpdir_factory.getbasetemp()
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks)
    unmod_schedule_file = tmpdir / "lightning_logs" / "version_0" / f"{model.__class__.__name__}_ft_schedule.yaml"
    with pytest.raises(SystemExit):
        trainer.fit(model)
    mod_sched_dict = trainer.finetuning_scheduler_callback.load_yaml_schedule(unmod_schedule_file)
    mod_sched_dict[0]["params"].extend(mod_sched_dict.pop(1)["params"])
    mod_sched_dict[0]["max_transition_epoch"] = 3
    mod_sched_dict[1] = mod_sched_dict.pop(2)
    mod_sched_dict[2] = mod_sched_dict.pop(3)
    mod_sched_dict[2]["params"] = ["layer.0.*"]
    epoch_only_sched = deepcopy(mod_sched_dict)
    epoch_only_sched[1]["max_transition_epoch"] = 2
    epoch_only_sched[2]["max_transition_epoch"] = 2
    return unmod_schedule_file, mod_sched_dict, epoch_only_sched


class ComplexNestedModel(LightningModule):
    """A nested model with a parent (non-leaf) module parameter to validate scheduled finetuning with such
    architectures."""

    def __init__(self):
        super().__init__()
        self.test = nn.Sequential(
            OrderedDict(
                [("encoder", nn.Sequential(ConvBlockParam(3, 64), ConvBlock(64, 128))), ("decoder", ConvBlock(128, 10))]
            )
        )

    def forward(self, x):
        return self.test(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [lr_scheduler]

    def training_step(self):
        pass

    def train_dataloader(self):
        pass


@pytest.mark.parametrize(
    "model, expected",
    [
        (FinetuningSchedulerBoringModel(), (4, ["layer.2.bias", "layer.2.weight"], ["layer.0.bias", "layer.0.weight"])),
        (ParityModuleRNN(), (3, ["rnn.bias_hh_l0", "rnn.bias_ih_l0"], ["rnn.weight_hh_l0", "rnn.weight_ih_l0"])),
        (
            ComplexNestedModel(),
            (7, ["test.decoder.conv.bias", "test.decoder.conv.weight"], ["test.encoder.0.parent_param"]),
        ),
    ],
    ids=["Boring", "ParityRNN", "ComplexNested"],
)
def test_gen_ft_schedule(tmpdir, model: "LightningModule", expected: Tuple):
    """Validate the default finetuning schedule generation."""
    seed_everything(42)
    callbacks = [FinetuningScheduler(gen_ft_sched_only=True)]
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks)
    ft_schedule = tmpdir / "lightning_logs" / "version_0" / f"{model.__class__.__name__}_ft_schedule.yaml"
    with pytest.raises(SystemExit):
        trainer.fit(model)
    seed_everything(42)
    assert os.path.isfile(ft_schedule)
    with open(ft_schedule) as f:
        test_schedule = yaml.safe_load(f.read())
    assert isinstance(test_schedule, Dict)
    assert len(test_schedule) == expected[0]
    assert test_schedule[1]["params"] == expected[1]
    assert test_schedule[next(reversed(list(test_schedule.keys())))]["params"] == expected[2]


EXPECTED_EXPIMP_RESULTS = {
    (True, -1): (5, 0, 2, 5, 8, 3, 3),
    (False, -1): (7, 0, 3, 7, 8, 4, 4),
    (True, 0): (4, 0, 0, 4, 4, 1, 1),
    (False, 0): (4, 0, 0, 4, 2, 1, 1),
    (True, 2): (5, 0, 2, 5, 8, 3, 3),
    (False, 2): (6, 0, 2, 6, 6, 3, 3),
    (True, 999): (5, 0, 2, 5, 8, 3, 3),
    (False, 999): (7, 0, 3, 7, 8, 4, 4),
}


@pytest.mark.parametrize("explicit_mode", [True, False], ids=["explicit", "implicit"])
@pytest.mark.parametrize("max_depth", [-1, 0, 2, 999], ids=["default", "maxdepth0", "maxdepth2", "maxdepth999"])
def test_finetuningscheduling_explicit_implicit(tmpdir, boring_ft_schedule, explicit_mode: bool, max_depth: int):
    """Validate scheduled finetuning works as expected in 'explicit' and 'implicit' modes in the context of various
    max_depth specifications."""
    seed_everything(42)
    ft_schedule = boring_ft_schedule[1] if explicit_mode else None
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=1),
        FTSCheckpoint(monitor="val_loss", verbose=True),
        FinetuningScheduler(ft_schedule=ft_schedule, max_depth=max_depth),
    ]
    model = FinetuningSchedulerBoringModel()
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks)
    trainer.fit(model)
    expected_state = EXPECTED_EXPIMP_RESULTS[(explicit_mode, max_depth)]
    assert trainer.early_stopping_callback.stopped_epoch == expected_state[0]
    assert trainer.finetuning_scheduler_callback.depth_remaining == expected_state[1]
    assert trainer.finetuning_scheduler_callback.curr_depth == expected_state[2]
    assert trainer.finetuning_scheduler_callback._fts_state._ft_epoch == expected_state[3]
    assert len(trainer.finetuning_scheduler_callback._fts_state._curr_thawed_params) == expected_state[4]
    assert len(trainer.finetuning_scheduler_callback._internal_optimizer_metadata[0]) == expected_state[5]
    assert len(trainer.optimizers[0].param_groups) == expected_state[6]
    for pg in range(expected_state[6]):
        assert trainer.optimizers[0].param_groups[pg]["params"][0].requires_grad
    still_frozen = [
        p
        for i, d in enumerate(trainer.finetuning_scheduler_callback.ft_schedule)
        if i > trainer.finetuning_scheduler_callback.max_depth
        for p in trainer.finetuning_scheduler_callback.ft_schedule[d]["params"]
    ]
    assert not any([p.requires_grad for n, p in trainer.model.named_parameters() if n in still_frozen])
    assert trainer.finetuning_scheduler_callback.curr_depth == trainer.finetuning_scheduler_callback.max_depth
    assert trainer.finetuning_scheduler_callback._fts_state._ft_epoch == trainer._fit_loop.current_epoch


EXPECTED_DECAY_RESULTS = {
    (True, False): (5, 0, 2, 5, 8, 3, 3, 1e-6),
    (True, True): (5, 0, 2, 5, 8, 5, 5, 0.0),
    (False, False): (7, 0, 3, 7, 8, 4, 4, 1e-6),
    (False, True): (7, 0, 3, 7, 8, 7, 7, 0.0),
}


@pytest.mark.parametrize("nodecay_mode", [False, True], ids=["alldecay", "nodecay"])
@pytest.mark.parametrize("explicit_mode", [True, False], ids=["explicit", "implicit"])
def test_finetuningscheduling_decay(tmpdir, boring_ft_schedule, explicit_mode: bool, nodecay_mode: bool):
    """Validate scheduled finetuning works as expected in 'explicit' and 'implicit' modes in the context of
    different nodecay list settings.

    Separately parameterized from :meth:`test_finetuningscheduling_explicit_implicit` to avoid
    costly increase in test volume w/ minimal benefit
    """
    seed_everything(42)
    ft_schedule = boring_ft_schedule[1] if explicit_mode else None
    no_decay = ["bias"] if nodecay_mode else None
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=1),
        FTSCheckpoint(monitor="val_loss", verbose=True),
        FinetuningScheduler(ft_schedule=ft_schedule, max_depth=-1),
    ]
    model = FinetuningSchedulerBoringModel(no_decay=no_decay)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks)
    trainer.fit(model)
    expected_state = EXPECTED_DECAY_RESULTS[(explicit_mode, nodecay_mode)]
    assert trainer.early_stopping_callback.stopped_epoch == expected_state[0]
    assert trainer.finetuning_scheduler_callback.depth_remaining == expected_state[1]
    assert trainer.finetuning_scheduler_callback.curr_depth == expected_state[2]
    assert trainer.finetuning_scheduler_callback._fts_state._ft_epoch == expected_state[3]
    assert len(trainer.finetuning_scheduler_callback._fts_state._curr_thawed_params) == expected_state[4]
    assert len(trainer.finetuning_scheduler_callback._internal_optimizer_metadata[0]) == expected_state[5]
    assert len(trainer.optimizers[0].param_groups) == expected_state[6]
    for pg in range(expected_state[6]):
        assert trainer.optimizers[0].param_groups[pg]["params"][0].requires_grad
    assert trainer.optimizers[0].param_groups[2]["weight_decay"] == expected_state[7]
    still_frozen = [
        p
        for i, d in enumerate(trainer.finetuning_scheduler_callback.ft_schedule)
        if i > trainer.finetuning_scheduler_callback.max_depth
        for p in trainer.finetuning_scheduler_callback.ft_schedule[d]["params"]
    ]
    assert not any([p.requires_grad for n, p in trainer.model.named_parameters() if n in still_frozen])
    assert trainer.finetuning_scheduler_callback.curr_depth == trainer.finetuning_scheduler_callback.max_depth
    assert trainer.finetuning_scheduler_callback._fts_state._ft_epoch == trainer._fit_loop.current_epoch


EXPECTED_RESUME_RESULTS = {
    ("best", False, -1): (0, 0, 3),
    ("kth", False, -1): (0, 0, 3),
    ("best", True, -1): (0, 0, 3),
    ("kth", True, -1): (1, 0, 3),
    ("best", False, 1): (0, 0, 1),
    ("kth", False, 1): (0, 0, 1),
    ("best", True, 1): (0, 0, 1),
    ("kth", True, 1): (1, 0, 1),
}


@pytest.mark.parametrize("ckpt,", ["best", "kth"], ids=["best", "kth"])
@pytest.mark.parametrize("inc_mode,", [False, True], ids=["defaultinc", "newinc"])
@pytest.mark.parametrize("max_depth", [-1, 1], ids=["nomaxdepth", "maxdepth1"])
def test_finetuningscheduler_callback_resume(tmpdir, ckpt_set, ckpt: str, inc_mode: bool, max_depth: int):
    """Validate scheduled finetuning resumption functions as expected from both 'best' and 'kth'(not-best)
    checkpoints in both new_incarnation modes with and without max_depth specified."""
    resume_callbacks = [
        EarlyStopping(monitor="val_loss", patience=1, min_delta=0.001),
        FTSCheckpoint(monitor="val_loss", verbose=True, save_top_k=3),
    ]
    resume_callbacks.append(FinetuningScheduler(new_incarnation_mode=inc_mode, max_depth=max_depth))

    seed_everything(42)
    model = FinetuningSchedulerBoringModel()
    trainer = Trainer(default_root_dir=tmpdir, callbacks=resume_callbacks)
    trainer.fit(model, ckpt_path=ckpt_set[ckpt])
    expected_state = EXPECTED_RESUME_RESULTS[(ckpt, getattr(resume_callbacks[2], "new_incarnation_mode"), max_depth)]
    assert trainer.checkpoint_callback.best_ckpt_depth == expected_state[0]
    assert trainer.finetuning_scheduler_callback.depth_remaining == expected_state[1]
    assert trainer.finetuning_scheduler_callback.curr_depth == expected_state[2]
    assert trainer.finetuning_scheduler_callback.curr_depth == trainer.finetuning_scheduler_callback.max_depth


EXPECTED_INTRAFIT_STATE = {
    0: (0, 3, 0, 0, 0, 0, 2, 1, 0, 0),
    1: (0, 3, 1, 0, 0, 1, 2, 1, 0, 0),
    2: (0, 3, 2, 0, 0, 1, 2, 1, 0, 0),
    3: (0, 3, 3, 0, 0, 1, 2, 1, 0, 0),
    4: (0, 3, 4, 0, 0, 1, 2, 1, 0, 0),
    5: (1, 2, 5, 0, 0, 1, 4, 2, 0, 0),
    6: (2, 1, 6, 0, 0, 1, 6, 3, 0, 0),
    7: (3, 0, 7, 0, 0, 1, 8, 4, 0, 0),
}


@pytest.mark.parametrize("restore_best", [True, False], ids=["default", "norestorebest"])
def test_finetuningscheduling_intrafit(tmpdir, restore_best: bool):
    """Inspect scheduled finetuning state within the training process to ensure it is taking the expected path in
    both restore_best modes."""
    seed_everything(42)
    model = FinetuningSchedulerBoringModel()
    callbacks = [
        TestFinetuningScheduler(expected_state=EXPECTED_INTRAFIT_STATE, restore_best=restore_best),
        EarlyStopping(monitor="val_loss", patience=1),
    ]
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks)
    trainer.fit(model)
    assert trainer.finetuning_scheduler_callback.depth_remaining == 0
    assert trainer.finetuning_scheduler_callback.curr_depth == 3
    assert trainer.finetuning_scheduler_callback.curr_depth == trainer.finetuning_scheduler_callback.max_depth


@pytest.mark.parametrize(
    "callbacks, expected",
    [
        ([FinetuningScheduler()], ("an EarlyStopping", "a finetuning")),
        ([FinetuningScheduler(), EarlyStopping(monitor="val_loss", patience=1)], ("a finetuning")),
        ([FinetuningScheduler(), FTSCheckpoint(monitor="val_loss", verbose=True)], ("an EarlyStopping")),
        (
            [
                FinetuningScheduler(epoch_transitions_only=True),
                FTSCheckpoint(monitor="val_loss", verbose=True),
                EarlyStopping(monitor="val_loss", patience=1),
            ],
            ("extraneous Early"),
        ),
    ],
    ids=["default", "nondef_es", "nondef_ftsckpt", "eponly_es"],
)
def test_finetuningscheduler_callback_warns(tmpdir, recwarn, callbacks: List[Callback], expected: Tuple[str]):
    """Validate :class:`~pytorch_lightning.callbacks.finetuning_scheduler.FinetuningScheduler` warnings that require a
    :class:`~pytorch_lighting.trainer.Trainer` to be defined are properly issued"""
    _ = Trainer(default_root_dir=tmpdir, callbacks=callbacks)
    assert all([any([re.compile(w_msg).search(w.message.args[0]) for w in recwarn.list]) for w_msg in expected])


def test_finetuningscheduling_opt_warns():
    """Validate :class:`~pytorch_lightning.callbacks.finetuning_scheduler.FinetuningScheduler` warnings that
    require only an :class:`~pytorch_lighting.optim.Optimizer` to be defined are properly issued."""
    fts = FinetuningScheduler()
    lm = FinetuningSchedulerBoringModel()
    opt = torch.optim.SGD(lm.parameters(), lr=1e-3)
    thawed_pl = []
    with pytest.warns(UserWarning, match="no new optimizer groups will be added"):
        fts.add_optimizer_groups(lm, opt, thawed_pl)


@pytest.mark.parametrize(
    "callbacks, expected",
    [
        ([FTSCheckpoint(monitor="val_loss", verbose=True)], "please use the standard ModelCheckpoint callback."),
        (
            [FinetuningScheduler(), FTSCheckpoint(monitor="val_loss", save_top_k=0)],
            "Please set save_top_k to a non-zero value",
        ),
        ([FinetuningScheduler(), FTSCheckpoint(monitor=None)], "but has no quantity to monitor"),
        ([FinetuningScheduler(ft_schedule="/tmp/fnf")], "Could not find specified finetuning scheduling file"),
    ],
    ids=["nofts", "topk0", "nomon", "schedfnf"],
)
def test_finetuningscheduling_misconfiguration(tmpdir, callbacks: List[Callback], expected: str):
    """Validate :class:`~pytorch_lightning.callbacks.finetuning_scheduler.FinetuningScheduler` misconfiguration
    exceptions are properly raised."""
    with pytest.raises(MisconfigurationException, match=expected):
        _ = Trainer(default_root_dir=tmpdir, callbacks=callbacks)
        fts = callbacks[0]
        if fts.ft_schedule:
            _ = fts.load_yaml_schedule(fts.ft_schedule)


@pytest.mark.parametrize(
    "strategy, gpus, plugins",
    [
        pytest.param("ddp2", 1, None, marks=RunIf(min_gpus=1)),
        pytest.param("ddp_fully_sharded", 1, None, marks=RunIf(min_gpus=1)),
        pytest.param("horovod", None, None, marks=RunIf(min_gpus=1)),
        pytest.param("deepspeed_stage_2", 1, None, marks=RunIf(deepspeed=True, min_gpus=1)),
    ],
)
def test_finetuningscheduling_distributed_compat(tmpdir, strategy, gpus, plugins):
    """Validate :class:`~pytorch_lightning.callbacks.finetuning_scheduler.FinetuningScheduler` misconfiguration
    exceptions are properly raised for currently unsupported plugins."""
    callbacks = [FinetuningScheduler()]
    with pytest.raises(MisconfigurationException, match="has not yet been adapted for the specified distributed"):
        _ = Trainer(default_root_dir=tmpdir, callbacks=callbacks, strategy=strategy, gpus=gpus, plugins=plugins)


def test_finetuningscheduling_optimizer_compat(tmpdir):
    """Validate :class:`~pytorch_lightning.callbacks.finetuning_scheduler.FinetuningScheduler` misconfiguration
    exceptions are properly raised for multi-optimizer configurations."""

    class MultiOptFTSBoringModel(FinetuningSchedulerBoringModel):
        def configure_optimizers(self):
            parameters = list(filter(lambda x: x.requires_grad, self.parameters()))
            optimizer0 = torch.optim.SGD(parameters, lr=1e-3)
            optimizer1 = torch.optim.SGD(parameters, lr=1e-3)
            return [optimizer0, optimizer1]

    seed_everything(42)
    model = MultiOptFTSBoringModel()
    callbacks = [FinetuningScheduler()]
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks)
    with pytest.raises(MisconfigurationException, match="single-optimizer configuration"):
        trainer.fit(model)


@pytest.mark.parametrize(
    "epoch_only_cfg, expected_state",
    [(True, ((0, 2, 5, 8, 3, 3), "maximum phase-specified")), (False, (None, "missing a max_"))],
    ids=["eponly", "noeponly"],
)
def test_finetuningscheduling_epoch_trans_only(tmpdir, boring_ft_schedule, epoch_only_cfg: bool, expected_state: Tuple):
    """Validate scheduled finetuning works as expected in 'epoch_transitions_only' mode while raising the
    appropriate exception/warning with respect to epoch_transitions_only scheduling and early stopping
    respectively."""
    seed_everything(42)
    # use appropriately configured epoch_transitions_only schedule if epoch_only_cfg, else validate config error thrown
    ft_schedule = boring_ft_schedule[2] if epoch_only_cfg else boring_ft_schedule[1]
    model = FinetuningSchedulerBoringModel()
    callbacks = [
        FTSCheckpoint(monitor="val_loss", verbose=True),
        FinetuningScheduler(ft_schedule=ft_schedule, epoch_transitions_only=True),
    ]
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks, max_epochs=6)
    if epoch_only_cfg:
        # we're testing an epoch_transitions_only schedule that should trigger the specified warning
        with pytest.warns(UserWarning, match=expected_state[1]):
            trainer.fit(model)
        # for the valid epoch_only_transitions schedule, verify expected state
        assert trainer.finetuning_scheduler_callback.depth_remaining == expected_state[0][0]
        assert trainer.finetuning_scheduler_callback.curr_depth == expected_state[0][1]
        assert trainer.finetuning_scheduler_callback._fts_state._ft_epoch == expected_state[0][2]
        assert len(trainer.finetuning_scheduler_callback._fts_state._curr_thawed_params) == expected_state[0][3]
        assert len(trainer.finetuning_scheduler_callback._internal_optimizer_metadata[0]) == expected_state[0][4]
        assert len(trainer.optimizers[0].param_groups) == expected_state[0][5]
        for pg in range(expected_state[0][5]):
            assert trainer.optimizers[0].param_groups[pg]["params"][0].requires_grad
        assert trainer.finetuning_scheduler_callback.curr_depth == trainer.finetuning_scheduler_callback.max_depth
        assert trainer.finetuning_scheduler_callback._fts_state._ft_epoch == trainer._fit_loop.current_epoch
    else:
        with pytest.raises(MisconfigurationException, match=expected_state[1]):
            trainer.fit(model)


@RunIf(special=True, min_gpus=2)
def test_fts_multi_dp(tmpdir):
    """Validate :class:`~pytorch_lightning.callbacks.finetuning_scheduler.FinetuningScheduler` functions properly
    in a supported 'dp' distributed context."""
    seed_everything(42)
    model = FinetuningSchedulerBoringModel()
    callbacks = [FinetuningScheduler(), EarlyStopping(monitor="val_loss", patience=1)]
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks, strategy="dp", gpus=2)
    trainer.fit(model)
    assert trainer.finetuning_scheduler_callback.depth_remaining == 0
    assert trainer.finetuning_scheduler_callback.curr_depth == 3
    assert trainer.finetuning_scheduler_callback.curr_depth == trainer.finetuning_scheduler_callback.max_depth


@RunIf(special=True, min_gpus=2)
def test_fts_multi_ddp(tmpdir):
    """Validate :class:`~pytorch_lightning.callbacks.finetuning_scheduler.FinetuningScheduler` functions properly
    in a supported 'ddp' distributed context."""
    seed_everything(42)
    model = FinetuningSchedulerBoringModel()
    callbacks = [FinetuningScheduler(), EarlyStopping(monitor="val_loss", patience=1)]
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks, strategy="ddp", gpus=2)
    trainer.fit(model)
    assert trainer.finetuning_scheduler_callback.depth_remaining == 0
    assert trainer.finetuning_scheduler_callback.curr_depth == 3
    assert trainer.finetuning_scheduler_callback.curr_depth == trainer.finetuning_scheduler_callback.max_depth


@RunIf(special=True, min_gpus=2)
def test_fts_multi_ddp_sharded(tmpdir):
    """Validate :class:`~pytorch_lightning.callbacks.finetuning_scheduler.FinetuningScheduler` functions properly
    in a supported 'ddp_sharded' distributed context."""
    seed_everything(42)
    model = FinetuningSchedulerBoringModel()
    callbacks = [FinetuningScheduler(), EarlyStopping(monitor="val_loss", patience=1)]
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks, strategy="ddp_sharded", gpus=2)
    trainer.fit(model)
    assert trainer.finetuning_scheduler_callback.depth_remaining == 0
    assert trainer.finetuning_scheduler_callback.curr_depth == 3
    assert trainer.finetuning_scheduler_callback.curr_depth == trainer.finetuning_scheduler_callback.max_depth


@RunIf(special=True, min_gpus=2)
def test_fts_multi_ddp_spawn(tmpdir):
    """Validate :class:`~pytorch_lightning.callbacks.finetuning_scheduler.FinetuningScheduler` functions properly
    in a supported 'ddp_spawn' distributed context."""
    seed_everything(42)
    model = FinetuningSchedulerBoringModel()
    callbacks = [FinetuningScheduler(), EarlyStopping(monitor="val_loss", patience=1)]
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks, strategy="ddp_spawn", gpus=2)
    trainer.fit(model)
    assert trainer.callback_metrics["val_loss"] < 0.1


@RunIf(special=True, min_gpus=2)
def test_fts_multi_ddp_sharded_spawn(tmpdir):
    """Validate :class:`~pytorch_lightning.callbacks.finetuning_scheduler.FinetuningScheduler` functions properly
    in a supported 'ddp_sharded_spawn' distributed context."""
    seed_everything(42)
    model = FinetuningSchedulerBoringModel()
    callbacks = [FinetuningScheduler(), EarlyStopping(monitor="val_loss", patience=1)]
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks, strategy="ddp_sharded_spawn", gpus=2)
    trainer.fit(model)
    assert trainer.callback_metrics["val_loss"] < 0.1
