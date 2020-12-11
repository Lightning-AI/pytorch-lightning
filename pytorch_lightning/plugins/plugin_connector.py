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
from enum import Enum
from typing import List, Optional, Union

from pytorch_lightning.cluster_environments import ClusterEnvironment
from pytorch_lightning.plugins.apex import ApexPlugin
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.plugins.native_amp import NativeAMPPlugin
from pytorch_lightning.plugins.plugin import LightningPlugin
from pytorch_lightning.plugins.sharded_plugin import DDPShardedPlugin
from pytorch_lightning.utilities import AMPType, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class PluginConnector:

    def __init__(self, trainer):
        self.trainer = trainer
        self.plugins = []
        self.ddp_plugin = DDPPlugin()
        self.cloud_environment = None
        self.amp_plugin = NativeAMPPlugin(trainer)
        self.apex_plugin = ApexPlugin(trainer)

    def on_trainer_init(self, plugins: Optional[Union[str, list]]):
        self.plugins = plugins
        if self.plugins is None:
            self.plugins = []
        self.plugins = self._convert_str_custom_plugins(self.plugins)
        self.plugins = self._append_required_plugins(self.plugins)
        self.__attach_ddp()
        self.__attach_cluster()
        self.__attach_amp()
        self.__attach_apex()

    def __attach_amp(self):
        amp_plugin = self.__attach_plugin(NativeAMPPlugin)
        if amp_plugin:
            self.trainer.amp_backend = AMPType.NATIVE
            self.trainer.precision_connector.backend = amp_plugin

    def __attach_apex(self):
        apex_plugin = self.__attach_plugin(ApexPlugin)
        if apex_plugin:
            self.trainer.amp_backend = AMPType.APEX
            self.trainer.precision_connector.backend = apex_plugin

    def __attach_plugin(self, plugin_type, limit=1):
        count = 0
        plugin_result = None
        for plugin in self.plugins:
            if isinstance(plugin, plugin_type):

                # count the clusters
                count += 1
                if count > limit:
                    m = f'you can only use one {plugin_type.__class__} in plugins. You passed in: {count}'
                    raise MisconfigurationException(m)

                plugin_result = plugin

        return plugin_result

    def __attach_ddp(self, limit=1):
        count = 0
        for plugin in self.plugins:
            if isinstance(plugin, DDPPlugin):

                # count the clusters
                count += 1
                if count > limit:
                    m = f'you can only use one DDP plugin in plugins. You passed in: {count}'
                    raise MisconfigurationException(m)

                # set the ddp plugin
                self.ddp_plugin = plugin

    def __attach_cluster(self, limit=1):
        num_clusters = 0
        for plugin in self.plugins:
            if isinstance(plugin, ClusterEnvironment):

                # count the clusters
                num_clusters += 1
                if num_clusters > limit:
                    m = f'you can only use one cluster environment in plugins. You passed in: {num_clusters}'
                    raise MisconfigurationException(m)

                # set the cluster
                self.cloud_environment = plugin

    def _convert_str_custom_plugins(self, plugins: Union[str, list]):
        """
        Converts string inputs to corresponding supported lightning plugins.
        Args:
            plugins: List of plugins or string to choose lightning plugin.

        Returns: List of plugins where strings are now plugins.
        """
        if isinstance(plugins, str):
            return [self._convert_str_to_plugin(plugins)]
        return [self._convert_str_to_plugin(plugin) for plugin in plugins]

    def _convert_str_to_plugin(self, plugin):
        if isinstance(plugin, str):
            if plugin not in LightningCustomPlugins.__members__:
                raise MisconfigurationException(
                    f" {plugin} is not a supported lightning custom plugin."
                    " If you're trying to pass a custom plugin, please pass this as an object to"
                    " Trainer(plugins=[MyPlugin()]."
                    f" Supported plugins as string input: {[e.name for e in LightningCustomPlugins]}."
                )
            plugin_cls = LightningCustomPlugins[plugin].value
            return plugin_cls(trainer=self.trainer)
        return plugin

    def _append_required_plugins(self, plugins: List[LightningPlugin]):
        """
        Allows custom plugins to define additional plugins. This is useful for when custom plugins
        need to enforce override of native amp/apex when they are enabled.

        Args:
            plugins: List of plugins

        Returns: List of plugins containing additional plugins if needed.

        Example::
            class MyPlugin(DDPPlugin):
                def required_plugins(self):
                    return [MyCustomAMPPlugin()]

            # Will automatically add the necessary AMP plugin
            trainer = Trainer(plugins=[MyPlugin()])

            # Crash as MyPlugin enforces custom AMP plugin
            trainer = Trainer(plugins=[MyPlugin(), NativeAMPPlugin()])

        """
        for plugin in plugins:
            required_plugins = plugin.required_plugins(amp_backend=self.trainer.amp_backend, trainer=self.trainer)
            if required_plugins:
                rank_zero_warn(
                    f'plugin {type(plugin)} has added additional required plugins as default:'
                    f' {[type(x) for x in required_plugins]}'
                    'Extend this plugin and override `required_plugins`'
                    'if this conflicts with your additional plugins.'
                )
                plugins += required_plugins
        return plugins

    @classmethod
    def available_plugins(cls):
        """
        List of all available plugins that can be string arguments to the trainer.
        Returns: List of all available plugins that are supported as string arguments.
        """
        return [e.name for e in LightningCustomPlugins]


class LightningCustomPlugins(Enum):
    """
    String support for custom lightning plugins.
    Allows easier access to custom lightning plugins from the command line.
    """
    ddp_sharded = DDPShardedPlugin
    native_amp = NativeAMPPlugin
    apex_amp = ApexPlugin
