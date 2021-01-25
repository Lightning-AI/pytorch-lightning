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
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.plugin_connector import LightningCustomPlugins, PluginConnector


def test_available_plugins_trainer():
    """ Test that available plugins return the correct list in the trainer. """
    plugins = Trainer.available_plugins()
    expected_plugins = [e.name for e in LightningCustomPlugins]
    assert plugins == expected_plugins


def test_available_plugins_connector():
    """ Test that available plugins return the correct list in the connector. """
    plugins = PluginConnector.available_plugins()
    expected_plugins = [e.name for e in LightningCustomPlugins]
    assert plugins == expected_plugins
