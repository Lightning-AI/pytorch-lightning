import os
import time
from typing import Any
from unittest import mock
from unittest.mock import PropertyMock

import pytest
import torch

import pytorch_lightning as pl
from pytorch_lightning.plugins.environments.lightning_environment import find_free_network_port
from pytorch_lightning.strategies import CollaborativeStrategy
from pytorch_lightning.strategies.collaborative import HiveMindScheduler
from pytorch_lightning.utilities import _HIVEMIND_AVAILABLE
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf

if _HIVEMIND_AVAILABLE:
    import hivemind


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
    {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor", "PORT": str(find_free_network_port())},
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
def test_optimizer_wrapped_args_parsed(caplog):
    class TestModel(BoringModel):
        def on_before_backward(self, loss: torch.Tensor) -> None:
            optimizer = self.trainer.optimizers[0]
            assert isinstance(optimizer, hivemind.Optimizer)

    model = TestModel()
    trainer = pl.Trainer(strategy=CollaborativeStrategy(target_batch_size=1), fast_dev_run=True)
    trainer.fit(model)


@RunIf(hivemind=True, min_gpus=1)
@mock.patch.dict(os.environ, {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor"}, clear=True)
def test_scaler_updated_precision_16(caplog):
    class TestModel(BoringModel):
        def on_before_backward(self, loss: torch.Tensor) -> None:
            optimizer = self.trainer.optimizers[0]
            assert isinstance(self.trainer.precision_plugin.scaler, hivemind.GradScaler)
            assert isinstance(optimizer, hivemind.Optimizer)

    model = TestModel()
    trainer = pl.Trainer(strategy=CollaborativeStrategy(target_batch_size=1), fast_dev_run=True, precision=16, gpus=1)
    trainer.fit(model)


@RunIf(hivemind=True)
@mock.patch.dict(os.environ, {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor"}, clear=True)
def test_scheduler_wrapped(caplog):
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
        "INITIAL_PEERS": "TEST_PEERS",
        "HOST": "TEST_HOST",
        "PORT": "1300",
        "ENDPOINT": "1",
        "PEER_ENDPOINT": "TEST_PEER_ENDPOINT",
    },
    clear=True,
)
@mock.patch("hivemind.DHT", autospec=True)
@mock.patch("lightning_collaborative.server.DHTManager._get_peers", autospec=True)
@mock.patch("http.server.ThreadingHTTPServer", autospec=True)
def test_env_variables_parsed(mock_dht, mock_peers, mock_server):
    strategy = CollaborativeStrategy(target_batch_size=1)
    assert strategy.dht_manager.initial_peers == ["TEST_PEERS"]
    assert strategy.dht_manager.host == "TEST_HOST"
    assert strategy.dht_manager.port == "1300"
    assert strategy.dht_manager.endpoint
    assert strategy.dht_manager.peer_endpoint == "TEST_PEER_ENDPOINT"


@RunIf(hivemind=True)
@mock.patch.dict(os.environ, {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor"}, clear=True)
@mock.patch("hivemind.Optimizer", wraps=hivemind.Optimizer)
@mock.patch("http.server.ThreadingHTTPServer", autospec=True)
@mock.patch("lightning_collaborative.strategy.CollaborativeStrategy.num_peers", new_callable=PropertyMock)
def test_args_passed_to_optimizer(mock_peers, mock_server, mock_optimizer):
    mock_peers.return_value = 1
    compression = hivemind.ScaledFloat16Compression()

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


@RunIf(hivemind=True)
@mock.patch.dict(os.environ, {"HIVEMIND_MEMORY_SHARING_STRATEGY": "file_descriptor"}, clear=True)
@pytest.mark.parametrize(
    "host_maddrs,expected_maddrs",
    [(None, ["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"]), (["/ip4/127.0.0.1/tcp/0"], ["/ip4/127.0.0.1/tcp/0"])],
)
def test_maddrs(host_maddrs, expected_maddrs):
    strategy = CollaborativeStrategy(target_batch_size=1, host_maddrs=host_maddrs)
    assert strategy.dht.kwargs["host_maddrs"] == expected_maddrs


def _test_endpoint_run_fn(endpoint, peer_endpoint, port):
    class TestModel(BoringModel):
        def on_train_batch_start(self, batch: Any, batch_idx: int, unused: int = 0):
            time.sleep(1)

    model = TestModel()
    trainer = pl.Trainer(
        max_epochs=1,
        strategy=CollaborativeStrategy(
            target_batch_size=128, endpoint=endpoint, peer_endpoint=peer_endpoint, port=port
        ),
    )
    trainer.fit(model)


# TODO: figure out how to do a multiple process test.
# @RunIf(hivemind=True, standalone=True)
# def test_endpoint_run():
#     port = find_free_network_port()
#     endpoint = f"0.0.0.0:{port}"
#     args = [(True, None, port), (False, endpoint, port)]
#     with ThreadPool(processes=2) as pool:
#         all_links = pool.starmap(_test_endpoint_run_fn, args)
