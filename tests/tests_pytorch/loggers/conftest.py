import sys
from types import ModuleType
from unittest.mock import Mock, MagicMock

import pytest


@pytest.fixture()
def mlflow_mock(monkeypatch):
    mlflow = ModuleType("mlflow")
    mlflow.set_tracking_uri = Mock()
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)

    mlflow_tracking = ModuleType("tracking")
    mlflow_tracking.MlflowClient = Mock()
    mlflow_tracking.artifact_utils = Mock()
    monkeypatch.setitem(sys.modules, "mlflow.tracking", mlflow_tracking)

    mlflow_entities = ModuleType("entities")
    mlflow_entities.Metric = Mock()
    mlflow_entities.Param = Mock()
    mlflow_entities.time = Mock()
    monkeypatch.setitem(sys.modules, "mlflow.entities", mlflow_entities)

    mlflow.tracking = mlflow_tracking
    mlflow.entities = mlflow_entities
    return mlflow


@pytest.fixture()
def wandb_mock(monkeypatch):
    wandb = ModuleType("wandb")
    # Run = Mock()
    wandb.init = MagicMock()
    wandb.run = Mock()
    wandb.require = Mock()
    wandb.Api = Mock()
    wandb.Artifact = Mock()
    wandb.Table = Mock()
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    wandb_sdk = ModuleType("sdk")
    monkeypatch.setitem(sys.modules, "wandb.sdk", wandb_sdk)

    wandb_sdk_lib = ModuleType("lib")
    wandb_sdk_lib.RunDisabled = Mock()
    monkeypatch.setitem(sys.modules, "wandb.sdk.lib", wandb_sdk_lib)

    wandb_wandb_run = ModuleType("wandb_run")
    wandb_wandb_run.Run = Mock()
    monkeypatch.setitem(sys.modules, "wandb.wandb_run", wandb_wandb_run)

    wandb.sdk = wandb_sdk
    wandb.sdk.lib = wandb_sdk_lib
    wandb.wandb_run = wandb_wandb_run
    return wandb
