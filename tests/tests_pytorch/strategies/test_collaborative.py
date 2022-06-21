import multiprocessing as mp
import os
import time
from typing import Any
from unittest import mock
from unittest.mock import PropertyMock

import pytest
import requests
import torch
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.plugins.environments.lightning_environment import find_free_network_port
from pytorch_lightning.strategies import CollaborativeStrategy
from pytorch_lightning.strategies.collaborative import HiveMindScheduler
from pytorch_lightning.utilities import _HIVEMIND_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tests_pytorch.helpers.runif import RunIf

if _HIVEMIND_AVAILABLE:
    import hivemind


@mock.patch("pytorch_lightning.strategies.collaborative._HIVEMIND_AVAILABLE", False)
def test_raise_exception_if_hivemind_unavailable():
    """Test that we raise an exception when Hivemind is not available."""
    with pytest.raises(MisconfigurationException, match="you must have Hivemind installed"):
        CollaborativeStrategy(target_batch_size=1)


@RunIf(hivemind=True)
@mock.patch("hivemind.DHT", autospec=True)
def test_strategy(mock_dht):
    strategy = CollaborativeStrategy(target_batch_size=1)
    trainer = pl.Trainer(strategy=strategy)
    assert trainer.strategy == strategy


@RunIf(hivemind=True)
@mock.patch("hivemind.DHT", autospec=True)
@mock.patch("pytorch_lightning.strategies.collaborative.DHTManager._get_peers", autospec=True)
@pytest.mark.parametrize(
    "initial_peers,peer_endpoint",
    [(["TEST"], None), (None, "localhost:153")],
)
def test_logging_disabled_when_second_peer(mock_dht, mock_http, initial_peers, peer_endpoint):
    """Test when we are a second peer (passing initial peers or peer endpoint) we warn the user that
    logging/checkpointing will be disabled."""
    with pytest.warns(UserWarning, match="This machine is not a persistent machine"):
        CollaborativeStrategy(target_batch_size=1, initial_peers=initial_peers, peer_endpoint=peer_endpoint)


@RunIf(hivemind=True)
@mock.patch.dict(
    os.environ,
    {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor", "PL_PORT": str(find_free_network_port())},
    clear=True,
)
@pytest.mark.parametrize(
    "endpoint,expected_message",
    [(False, "INITIAL_PEERS"), (True, "Sidecar endpoint enabled to serve peers.")],
)
def test_initial_peer_message(caplog, endpoint, expected_message):
    model = BoringModel()
    trainer = pl.Trainer(strategy=CollaborativeStrategy(target_batch_size=1, endpoint=endpoint), fast_dev_run=True)
    trainer.fit(model)
    assert expected_message in caplog.text


@RunIf(hivemind=True)
@mock.patch.dict(os.environ, {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor"}, clear=True)
def test_optimizer_wrapped():
    class TestModel(BoringModel):
        def on_before_backward(self, loss: torch.Tensor) -> None:
            optimizer = self.trainer.optimizers[0]
            assert isinstance(optimizer, hivemind.Optimizer)

    model = TestModel()
    trainer = pl.Trainer(strategy=CollaborativeStrategy(target_batch_size=1), fast_dev_run=True)
    trainer.fit(model)


@RunIf(hivemind=True)
@mock.patch.dict(os.environ, {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor"}, clear=True)
def test_scheduler_wrapped():
    class TestModel(BoringModel):
        def on_before_backward(self, loss: torch.Tensor) -> None:
            scheduler = self.trainer.lr_scheduler_configs[0].scheduler
            assert isinstance(scheduler, HiveMindScheduler)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return [optimizer], [torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)]

    model = TestModel()
    trainer = pl.Trainer(
        strategy=CollaborativeStrategy(target_batch_size=1),
        fast_dev_run=True,
    )
    trainer.fit(model)


@RunIf(hivemind=True)
@mock.patch.dict(
    os.environ,
    {
        "HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor",
        "PL_INITIAL_PEERS": "TEST_PEERS",
        "PL_HOST": "TEST_HOST",
        "PL_PORT": "1300",
        "PL_ENDPOINT": "1",
        "PL_PEER_ENDPOINT": "TEST_PEER_ENDPOINT",
    },
    clear=True,
)
@mock.patch("hivemind.DHT", autospec=True)
@mock.patch("pytorch_lightning.strategies.collaborative.DHTManager._get_peers", autospec=True)
@mock.patch("http.server.ThreadingHTTPServer", autospec=True)
def test_env_variables_parsed(mock_dht, mock_peers, mock_server):
    """Test that env variables are parsed correctly."""
    strategy = CollaborativeStrategy(target_batch_size=1)
    assert strategy.dht_manager._initial_peers == ["TEST_PEERS"]
    assert strategy.dht_manager._host == "TEST_HOST"
    assert strategy.dht_manager._port == 1300
    assert strategy.dht_manager._endpoint
    assert strategy.dht_manager._peer_endpoint == "TEST_PEER_ENDPOINT"


@RunIf(hivemind=True)
@mock.patch.dict(os.environ, {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor"}, clear=True)
def test_reuse_grad_buffers_warning():
    """Test to ensure we warn when a user overrides `optimizer_zero_grad` and `reuse_grad_buffers` is True."""

    class TestModel(BoringModel):
        def on_before_backward(self, loss: torch.Tensor) -> None:
            optimizer = self.trainer.optimizers[0]
            assert isinstance(optimizer, hivemind.Optimizer)

        def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
            pass

    model = TestModel()
    trainer = pl.Trainer(
        strategy=CollaborativeStrategy(target_batch_size=1, reuse_grad_buffers=True), fast_dev_run=True
    )

    with pytest.warns(UserWarning, match="You have overridden `optimizer_zero_grad` which will be disabled."):
        trainer.fit(model)


@RunIf(hivemind=True)
def test_raise_exception_multiple_optimizers():
    """Test that we raise an exception when multiple optimizers are provided."""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer, optimizer], [lr_scheduler]

    model = TestModel()
    trainer = pl.Trainer(strategy=CollaborativeStrategy(target_batch_size=1), fast_dev_run=True)

    with pytest.raises(MisconfigurationException, match="Hivemind only supports training with one optimizer."):
        trainer.fit(model)


@RunIf(hivemind=True)
@mock.patch("pytorch_lightning.utilities.data._extract_batch_size", autospec=True, return_value=[None])
def test_raise_exception_no_batch_size(mock_extract_batch_size):
    """Test that we raise an exception when no batch size is automatically found."""

    model = BoringModel()
    trainer = pl.Trainer(strategy=CollaborativeStrategy(target_batch_size=1), fast_dev_run=True)

    with pytest.raises(MisconfigurationException, match="Please provide the batch size to the Strategy."):
        trainer.fit(model)


@RunIf(hivemind=True)
@mock.patch.dict(os.environ, {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor"}, clear=True)
@pytest.mark.parametrize(
    "delay_grad_averaging, delay_state_averaging, delay_optimizer_step",
    [(True, True, True), (False, True, False)],
)
def test_warn_if_argument_passed(delay_grad_averaging, delay_state_averaging, delay_optimizer_step):
    """Test ensures that valid combination of HiveMind delay arguments warn if scheduler isn't passed in as a
    function."""
    model = BoringModel()
    trainer = pl.Trainer(
        strategy=CollaborativeStrategy(
            target_batch_size=1,
            delay_grad_averaging=delay_grad_averaging,
            delay_state_averaging=delay_state_averaging,
            delay_optimizer_step=delay_optimizer_step,
        ),
        fast_dev_run=True,
    )

    with pytest.warns(UserWarning, match="requires a `scheduler_fn` to be passed to the strategy"):
        trainer.fit(model)


@RunIf(hivemind=True)
@mock.patch.dict(os.environ, {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor"}, clear=True)
@mock.patch("http.server.ThreadingHTTPServer", autospec=True)
@mock.patch("pytorch_lightning.strategies.collaborative.CollaborativeStrategy.num_peers", new_callable=PropertyMock)
def test_args_passed_to_optimizer(mock_peers, mock_server):
    """Test to ensure arguments are correctly passed to the hivemind optimizer wrapper."""
    mock_peers.return_value = 1
    compression = hivemind.ScaledFloat16Compression()
    with mock.patch("hivemind.Optimizer", wraps=hivemind.Optimizer) as mock_optimizer:

        class TestModel(BoringModel):
            def on_before_backward(self, loss: torch.Tensor) -> None:
                args, kwargs = mock_optimizer.call_args
                mock_optimizer.assert_called()
                arguments = dict(
                    delay_optimizer_step=True,
                    delay_state_averaging=True,
                    state_averaging_compression=compression,
                    grad_compression=compression,
                    offload_optimizer=True,
                    reuse_grad_buffers=True,
                    target_batch_size=1,
                )

                for key, value in arguments.items():
                    assert key in kwargs
                    assert value == kwargs[key]

        model = TestModel()
        trainer = pl.Trainer(
            strategy=CollaborativeStrategy(
                target_batch_size=1,
                reuse_grad_buffers=True,
                delay_state_averaging=True,
                delay_optimizer_step=True,
                offload_optimizer=True,
                grad_compression=compression,
                state_averaging_compression=compression,
            ),
            fast_dev_run=True,
        )
        trainer.fit(model)
        # ensures that after training with `reuse_grad_buffers` we restore the hook
        assert model.optimizer_zero_grad is not None


@RunIf(hivemind=True)
@mock.patch.dict(os.environ, {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor"}, clear=True)
@pytest.mark.parametrize(
    "host_maddrs,expected_maddrs",
    [(None, ["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"]), (["/ip4/127.0.0.1/tcp/0"], ["/ip4/127.0.0.1/tcp/0"])],
)
def test_maddrs(host_maddrs, expected_maddrs):
    """Test that the multiple addresses are correctly assigned."""
    strategy = CollaborativeStrategy(target_batch_size=1, host_maddrs=host_maddrs)
    assert strategy.dht.kwargs["host_maddrs"] == expected_maddrs


def _run_collab_training_fn(initial_peers, wait_seconds, barrier, recorded_process_peers, recorded_process_steps):
    recorded_peers = []
    recorded_global_steps = []

    class TestModel(BoringModel):
        def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: int = 0) -> None:
            time.sleep(wait_seconds)  # add an additional delay to give processes time to sync
            recorded_peers.append(self.trainer.strategy.num_peers)
            recorded_global_steps.append(self.trainer.optimizers[0].local_epoch)

        def on_train_end(self) -> None:
            # wait for all processes to get to the end of training before teardown
            barrier.wait()

    model = TestModel()
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=16,
        limit_val_batches=0,
        strategy=CollaborativeStrategy(
            delay_state_averaging=True,
            offload_optimizer=True,
            delay_optimizer_step=True,
            delay_grad_averaging=True,
            target_batch_size=8,
            initial_peers=initial_peers,
            verbose=False,
        ),
    )
    trainer.fit(model)

    recorded_process_peers.append(recorded_peers)
    recorded_process_steps.append(recorded_global_steps)


@RunIf(hivemind=True)
@mock.patch.dict(os.environ, {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor"}, clear=True)
@pytest.mark.parametrize(
    "num_processes, wait_seconds",
    [(2, 0.25)],
)
def test_multiple_peers(num_processes, wait_seconds):
    """Test to ensure that if we have two running processes with the same peers, they connect and train
    successfully."""
    dht_root = hivemind.DHT(start=True)
    barrier = mp.Barrier(num_processes)
    initial_peers = dht_root.get_visible_maddrs()

    with mp.Manager() as manager:
        # allows processes to return their recorded logged peers/steps
        recorded_process_peers = manager.list()
        recorded_process_steps = manager.list()
        processes = [
            mp.Process(
                target=_run_collab_training_fn,
                kwargs=dict(
                    initial_peers=initial_peers,
                    wait_seconds=wait_seconds,
                    barrier=barrier,
                    recorded_process_peers=recorded_process_peers,
                    recorded_process_steps=recorded_process_steps,
                ),
            )
            for x in range(num_processes)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        # assert that peers increase as expected and we run at-least 1 global step.
        for process_peers, process_steps in zip(recorded_process_peers, recorded_process_steps):
            assert any(num_peer == num_processes for num_peer in process_peers)
            assert any(global_step > 0 for global_step in process_steps)


@RunIf(hivemind=True, min_cuda_gpus=1)
@mock.patch.dict(os.environ, {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor"}, clear=True)
def test_scaler_updated_precision_16():
    class TestModel(BoringModel):
        def on_fit_start(self) -> None:
            assert isinstance(self.trainer.precision_plugin.scaler, hivemind.GradScaler)
            raise SystemExit

    model = TestModel()
    trainer = pl.Trainer(
        strategy=CollaborativeStrategy(target_batch_size=1),
        fast_dev_run=True,
        precision=16,
        accelerator="gpu",
        devices=1,
    )
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(hivemind=True)
def test_raise_when_peer_endpoint_unsuccessful(caplog):
    port = find_free_network_port()
    with pytest.raises(MisconfigurationException, match="Unable to get peers"):
        with mock.patch("requests.get", wraps=requests.get) as requests_mock:
            CollaborativeStrategy(
                target_batch_size=1,
                peer_endpoint=f"localhost:{port}",
                retry_endpoint_attempts=10,
                retry_endpoint_sleep_duration=0,
            )
    assert "Failed to get peers, retrying" in caplog.text
    assert requests_mock.call_count == 10
