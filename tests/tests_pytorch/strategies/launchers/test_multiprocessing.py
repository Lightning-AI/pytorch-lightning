# Copyright The Lightning AI team.
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
from multiprocessing import Process
from unittest import mock
from unittest.mock import ANY, Mock, call, patch

import pytest
import torch
from lightning.fabric.plugins import ClusterEnvironment
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies.launchers.multiprocessing import _GlobalStateSnapshot, _MultiProcessingLauncher
from lightning.pytorch.trainer.states import TrainerFn

from tests_pytorch.helpers.runif import RunIf


@mock.patch("lightning.pytorch.strategies.launchers.multiprocessing.mp.get_all_start_methods", return_value=[])
def test_multiprocessing_launcher_forking_on_unsupported_platform(_):
    with pytest.raises(ValueError, match="The start method 'fork' is not available on this platform"):
        _MultiProcessingLauncher(strategy=Mock(), start_method="fork")


@pytest.mark.parametrize("start_method", ["spawn", pytest.param("fork", marks=RunIf(standalone=True))])
@mock.patch("lightning.pytorch.strategies.launchers.multiprocessing.mp")
def test_multiprocessing_launcher_start_method(mp_mock, start_method):
    mp_mock.get_all_start_methods.return_value = [start_method]
    launcher = _MultiProcessingLauncher(strategy=Mock(), start_method=start_method)
    launcher.launch(function=Mock())
    mp_mock.get_context.assert_called_with(start_method)
    mp_mock.start_processes.assert_called_with(
        ANY,
        args=ANY,
        nprocs=ANY,
        start_method=start_method,
        join=False,
    )


@pytest.mark.parametrize("start_method", ["spawn", pytest.param("fork", marks=RunIf(standalone=True))])
@mock.patch("lightning.pytorch.strategies.launchers.multiprocessing.mp")
def test_multiprocessing_launcher_restore_globals(mp_mock, start_method):
    """Test that we pass the global state snapshot to the worker function only if we are starting with 'spawn'."""
    mp_mock.get_all_start_methods.return_value = [start_method]
    launcher = _MultiProcessingLauncher(strategy=Mock(), start_method=start_method)
    launcher.launch(function=Mock())
    function_args = mp_mock.start_processes.call_args[1]["args"]
    if start_method == "spawn":
        assert len(function_args) == 6
        assert isinstance(function_args[5], _GlobalStateSnapshot)
    else:
        assert len(function_args) == 5


def test_global_state_snapshot():
    """Test the capture() and restore() methods for the global state snapshot."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(123)

    # capture the state of globals
    snapshot = _GlobalStateSnapshot.capture()

    # simulate there is a process boundary and flags get reset here
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(321)

    # restore the state of globals
    snapshot.restore()
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark
    assert torch.initial_seed() == 123


@pytest.mark.parametrize("trainer_fn", [TrainerFn.FITTING, "other"])
@pytest.mark.parametrize("fake_node_rank", [0, 1])
@pytest.mark.parametrize("fake_local_rank", [0, 1])
def test_collect_rank_zero_results(trainer_fn, fake_node_rank, fake_local_rank, tmp_path):
    """Tests that the spawn strategy transfers the new weights to the main process and deletes the temporary file."""
    model = Mock(wraps=BoringModel(), spec=BoringModel)
    fake_global_rank = 2 * fake_node_rank + fake_local_rank

    cluster_environment = Mock(spec=ClusterEnvironment)
    cluster_environment.world_size.return_value = 4
    cluster_environment.node_rank.return_value = fake_node_rank
    cluster_environment.local_rank.return_value = fake_local_rank
    cluster_environment.global_rank.return_value = fake_global_rank

    strategy = DDPStrategy(cluster_environment=cluster_environment, start_method="spawn")
    strategy._local_rank = fake_local_rank

    launcher = _MultiProcessingLauncher(strategy=strategy)
    trainer = Trainer(accelerator="cpu", default_root_dir=tmp_path, strategy=strategy)

    assert strategy.node_rank == fake_node_rank
    assert strategy.local_rank == fake_local_rank
    assert strategy.global_rank == fake_global_rank

    trainer.strategy.connect(model)
    trainer.state.fn = trainer_fn  # pretend we are in a particular trainer state

    spawn_output = launcher._collect_rank_zero_results(trainer, {})

    model.state_dict.assert_called_once()
    is_fitting = trainer_fn == TrainerFn.FITTING
    if strategy.local_rank == 0:
        # on local rank 0 (each node), we expect a temp checkpoint (when fitting)
        assert not is_fitting or spawn_output.weights_path.endswith(".temp.ckpt")
        assert not is_fitting or os.path.isfile(spawn_output.weights_path)
        assert is_fitting or spawn_output.weights_path is None
    else:
        # all other ranks don't have outputs (rank 0 needs to handle the output)
        assert spawn_output is None


@pytest.mark.parametrize("trainer_fn", [TrainerFn.FITTING, "other"])
def test_transfer_weights(tmp_path, trainer_fn):
    """Tests that the multiprocessing launcher transfers the new weights to the main process and deletes the temporary
    file."""
    model = Mock(wraps=BoringModel(), spec=BoringModel)
    strategy = DDPStrategy(start_method="spawn")
    trainer = Trainer(accelerator="cpu", default_root_dir=tmp_path, strategy=strategy)
    trainer.strategy.connect(model)
    trainer.state.fn = trainer_fn  # pretend we are in a particular trainer state

    spawn_output = strategy._launcher._collect_rank_zero_results(trainer, {})

    model.state_dict.assert_called_once()
    if trainer_fn == TrainerFn.FITTING:
        assert spawn_output.weights_path.endswith(".temp.ckpt")
        assert os.path.isfile(spawn_output.weights_path)
    else:
        assert spawn_output.weights_path is None

    # <-- here would normally be the multiprocessing boundary
    strategy._launcher._recover_results_in_main_process(spawn_output, trainer)
    assert model.load_state_dict.call_count == int(spawn_output.weights_path is not None)


def test_non_strict_loading(tmp_path):
    """Tests that the multiprocessing launcher loads the weights back into the main process but with strict loading
    disabled, not erroring for missing keys."""
    model = Mock(wraps=BoringModel(), spec=BoringModel)
    strategy = DDPStrategy(start_method="spawn")
    trainer = Trainer(accelerator="cpu", default_root_dir=tmp_path, strategy=strategy)
    trainer.strategy.connect(model)
    trainer.state.fn = TrainerFn.FITTING  # state dict loading only relevant for the FITTING case

    spawn_output = strategy._launcher._collect_rank_zero_results(trainer, {})
    # <-- here would normally be the multiprocessing boundary
    strategy._launcher._recover_results_in_main_process(spawn_output, trainer)
    model.load_state_dict.assert_called_once_with(ANY, strict=False)


def test_kill():
    launcher = _MultiProcessingLauncher(Mock())
    proc0 = Mock(autospec=Process)
    proc1 = Mock(autospec=Process)
    launcher.procs = [proc0, proc1]

    with patch("os.kill") as kill_patch:
        launcher.kill(15)
    assert kill_patch.mock_calls == [call(proc0.pid, 15), call(proc1.pid, 15)]


class SimpleModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.tied_layer = torch.nn.Linear(32, 2)
        self.tied_layer.weight = self.layer.weight
        self.register_buffer("buffer", torch.ones(3))

    def on_fit_start(self) -> None:
        assert not self.layer.weight.is_shared()
        assert not self.tied_layer.weight.is_shared()
        assert not self.buffer.is_shared()

        # weights remain tied
        assert self.layer.weight.data_ptr() == self.tied_layer.weight.data_ptr()
        assert torch.equal(self.layer.weight.data, self.tied_layer.weight.data)


def test_memory_sharing_disabled(tmp_path):
    """Test that the multiprocessing launcher disables memory sharing on model parameters and buffers to avoid race
    conditions on model updates."""
    model = SimpleModel()
    assert not model.layer.weight.is_shared()
    assert model.layer.weight.data_ptr() == model.tied_layer.weight.data_ptr()

    trainer = Trainer(
        default_root_dir=tmp_path, logger=False, accelerator="cpu", devices=2, strategy="ddp_spawn", max_steps=0
    )
    trainer.fit(model)


def test_check_for_missing_main_guard():
    launcher = _MultiProcessingLauncher(strategy=Mock(), start_method="spawn")
    with mock.patch(
        "lightning.pytorch.strategies.launchers.multiprocessing.mp.current_process",
        return_value=Mock(_inheriting=True),  # pretend that main is importing itself
    ), pytest.raises(RuntimeError, match="requires that your script guards the main"):
        launcher.launch(function=Mock())


def test_fit_twice_raises(mps_count_0):
    model = BoringModel()
    trainer = Trainer(
        limit_train_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        strategy="ddp_spawn",
        barebones=True,
    )
    trainer.fit(model)
    trainer.test(model)  # make sure testing in between doesnt impact the result
    trainer.fit_loop.max_epochs += 1
    with pytest.raises(NotImplementedError, match=r"twice.*is not supported"):
        trainer.fit(model)
