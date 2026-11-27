from unittest.mock import MagicMock

import pytest
import requests
import torch
from torch import Tensor

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.serve.servable_module_validator import ServableModule, ServableModuleValidator
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import _xfail_gloo_windows


class ServableBoringModel(BoringModel, ServableModule):
    def configure_payload(self):
        return {"body": {"x": list(range(32))}}

    def configure_serialization(self):
        def deserialize(x):
            return torch.tensor(x, dtype=torch.float)

        def serialize(x):
            return x.tolist()

        return {"x": deserialize}, {"output": serialize}

    def serve_step(self, x: Tensor) -> dict[str, Tensor]:
        assert torch.equal(x, torch.arange(32, dtype=torch.float))
        return {"output": torch.tensor([0, 1])}

    def configure_response(self):
        return {"output": [0, 1]}


@pytest.mark.flaky(reruns=3)
def test_servable_module_validator():
    model = ServableBoringModel()
    callback = ServableModuleValidator()
    callback.on_train_start(Trainer(accelerator="cpu"), model)


@_xfail_gloo_windows
@pytest.mark.flaky(reruns=3)
def test_servable_module_validator_with_trainer(tmp_path, mps_count_0):
    callback = ServableModuleValidator()
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        callbacks=[callback],
        strategy="ddp_spawn",
        devices=2,
    )
    trainer.fit(ServableBoringModel())


def test_servable_module_validator_successful_on_error_response():
    validator = ServableModuleValidator()
    assert validator.successful is None

    resp_500 = requests.Response()
    resp_500.status_code = 500
    validator.resp = resp_500
    assert validator.successful is False
    assert validator.state_dict()["successful"] is False

    resp_200 = requests.Response()
    resp_200.status_code = 200
    validator.resp = resp_200
    assert validator.successful is True
    assert validator.state_dict()["successful"] is True


def test_servable_module_validator_cleans_up_process_on_invalid_payload(monkeypatch):
    class InvalidPayloadModel(ServableBoringModel):
        def configure_payload(self):
            return {"missing_body": {}}

    mock_process = MagicMock()
    monkeypatch.setattr("lightning.pytorch.serve.servable_module_validator.Process", MagicMock(return_value=mock_process))
    ping_resp = requests.Response()
    ping_resp.status_code = 200
    monkeypatch.setattr("lightning.pytorch.serve.servable_module_validator.requests.get", MagicMock(return_value=ping_resp))

    model = InvalidPayloadModel()
    trainer = Trainer(accelerator="cpu")
    model.trainer = trainer
    validator = ServableModuleValidator()

    with pytest.raises(Exception, match='should have a field named "body"'):
        validator.on_train_start(trainer, model)

    assert model.trainer is trainer
    mock_process.kill.assert_called_once()
    mock_process.join.assert_called_once()


def test_servable_module_validator_restores_trainer_if_process_start_fails(monkeypatch):
    mock_process = MagicMock()
    mock_process.start.side_effect = RuntimeError("process start failed")
    monkeypatch.setattr("lightning.pytorch.serve.servable_module_validator.Process", MagicMock(return_value=mock_process))

    model = ServableBoringModel()
    trainer = Trainer(accelerator="cpu")
    model.trainer = trainer
    validator = ServableModuleValidator()

    with pytest.raises(RuntimeError, match="process start failed"):
        validator.on_train_start(trainer, model)

    assert model.trainer is trainer


def test_servable_module_validator_non_200_response(monkeypatch):
    mock_process = MagicMock()
    monkeypatch.setattr("lightning.pytorch.serve.servable_module_validator.Process", MagicMock(return_value=mock_process))
    ping_resp = requests.Response()
    ping_resp.status_code = 200
    monkeypatch.setattr("lightning.pytorch.serve.servable_module_validator.requests.get", MagicMock(return_value=ping_resp))
    serve_resp = requests.Response()
    serve_resp.status_code = 500
    monkeypatch.setattr("lightning.pytorch.serve.servable_module_validator.requests.post", MagicMock(return_value=serve_resp))

    model = ServableBoringModel()
    trainer = Trainer(accelerator="cpu")
    validator = ServableModuleValidator(exit_on_failure=False)
    validator.on_train_start(trainer, model)

    assert validator.successful is False
    mock_process.kill.assert_called_once()
    mock_process.join.assert_called_once()

    validator_strict = ServableModuleValidator(exit_on_failure=True)
    with pytest.raises(MisconfigurationException, match="The model isn't servable"):
        validator_strict.on_train_start(trainer, model)

