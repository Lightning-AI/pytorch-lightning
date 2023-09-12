import sys
from types import ModuleType
from unittest.mock import Mock

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
def comet_mock(monkeypatch):
    comet = ModuleType("comet_ml")
    monkeypatch.setitem(sys.modules, "comet_ml", comet)

    comet.Experiment = Mock()
    comet.ExistingExperiment = Mock()
    comet.OfflineExperiment = Mock()
    comet.API = Mock()
    comet.config = Mock()

    comet_api = ModuleType("api")
    comet_api.API = Mock()
    monkeypatch.setitem(sys.modules, "comet_ml.api", comet_api)

    comet.api = comet_api
    return comet
