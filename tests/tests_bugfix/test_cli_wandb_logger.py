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
from unittest import mock

import pytest
import yaml
import glob

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers.wandb import WandbLogger

@pytest.fixture()
def cleandir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return


@pytest.fixture(autouse=True)
def ensure_cleandir():
    yield
    # make sure tests don't leave configuration files
    assert not glob.glob("*.yaml")


@pytest.mark.parametrize("log_model", [["True", True], ["False", False], ["all", "all"]])
def test_wandb_logger_set_log_model_param_via_cli(log_model, cleandir):
    """
    Ensure that setting the log_model parameter via CLI works as intended.
    Before the bug fix, this test fails, due to the wrong order in the Union type hint. 
    """

    passed_value, intended_value = log_model

    class LCLI_CheckLogModel(LightningCLI):
        def before_instantiate_classes(self):
            interpreted_value = self.config.trainer.logger.init_args.log_model
            assert type(interpreted_value) == type(intended_value)
            assert interpreted_value == intended_value

    # Create a config file with the log_model parameter set. This seems necessary to be able
    # to set the init_args parameter of the logger on the CLI later on.
    input_config = {
        "trainer": {
            "logger": {
                "class_path": "lightning.pytorch.loggers.wandb.WandbLogger",
                "init_args": {
                    "log_model": passed_value
                }
            },
        }
    }
    # Might be better to use a different file name for each different input parameter.
    config_path = "config.yaml"
    with open(config_path, "w") as f:
        f.write(yaml.dump(input_config))

    # Test case 1: Set the log_model parameter only via the config file.
    with mock.patch("sys.argv", ["any.py", "--config", config_path]):
        cli = LCLI_CheckLogModel(
            BoringModel, run=False, save_config_callback=None)

    # Test case 2: Overwrite the log_model parameter via the command line. This behaves incorrectly before the bug fix.
    wandb_cli_arg = f"--trainer.logger.init_args.log_model={log_model[0]}"

    with mock.patch("sys.argv", ["any.py", "--config", config_path, wandb_cli_arg]):
        cli = LCLI_CheckLogModel(
            BoringModel, run=False, save_config_callback=None)
        
    # Remove the config file.
    os.remove(config_path)