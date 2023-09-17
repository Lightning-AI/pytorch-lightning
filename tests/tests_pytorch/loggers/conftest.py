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
    class RunType:  # to make isinstance checks pass
        pass

    run_mock = Mock(
        spec=RunType, log=Mock(), config=Mock(), watch=Mock(), log_artifact=Mock(), use_artifact=Mock(), id="run_id"
    )

    wandb = ModuleType("wandb")
    wandb.init = Mock(return_value=run_mock)
    wandb.run = Mock()
    wandb.require = Mock()
    wandb.Api = Mock()
    wandb.Artifact = Mock()
    wandb.Image = Mock()
    wandb.Table = Mock()
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    wandb_sdk = ModuleType("sdk")
    monkeypatch.setitem(sys.modules, "wandb.sdk", wandb_sdk)

    wandb_sdk_lib = ModuleType("lib")
    wandb_sdk_lib.RunDisabled = RunType
    monkeypatch.setitem(sys.modules, "wandb.sdk.lib", wandb_sdk_lib)

    wandb_wandb_run = ModuleType("wandb_run")
    wandb_wandb_run.Run = RunType
    monkeypatch.setitem(sys.modules, "wandb.wandb_run", wandb_wandb_run)

    wandb.sdk = wandb_sdk
    wandb.sdk.lib = wandb_sdk_lib
    wandb.wandb_run = wandb_wandb_run
    return wandb


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


@pytest.fixture()
def neptune_mock(monkeypatch):
    class HandlerType:  # to make isinstance checks pass
        def get_root_object(self):
            pass

        def __getitem__(self, item):
            pass

        def __setitem__(self, key, value):
            pass

    handler_mock = MagicMock(spec=HandlerType, exists=Mock(), wait=Mock())
    handler_mock.get_root_object.return_value = handler_mock

    neptune = ModuleType("neptune")
    neptune.init_run = Mock(return_value=handler_mock)
    neptune.Run = HandlerType
    monkeypatch.setitem(sys.modules, "neptune", neptune)

    neptune_handler = ModuleType("handler")
    neptune_handler.Handler = HandlerType
    monkeypatch.setitem(sys.modules, "neptune.handler", neptune_handler)

    neptune_types = ModuleType("types")
    neptune_types.File = Mock()
    monkeypatch.setitem(sys.modules, "neptune.types", neptune_types)

    neptune_utils = ModuleType("utils")
    neptune_utils.stringify_unsupported = Mock()
    monkeypatch.setitem(sys.modules, "neptune.utils", neptune_utils)

    neptune.handler = neptune_handler
    neptune.types = neptune_types
    neptune.utils = neptune_utils
    return neptune
