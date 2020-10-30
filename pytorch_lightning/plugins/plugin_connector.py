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
from pytorch_lightning.cluster_environments import ClusterEnvironment
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.plugins.apex import ApexPlugin
from pytorch_lightning.plugins.native_amp import NativeAMPPlugin
from pytorch_lightning.utilities import AMPType


class PluginConnector:

    def __init__(self, trainer):
        self.trainer = trainer
        self.plugins = []
        self.ddp_plugin = DDPPlugin()
        self.cloud_environment = None
        self.amp_plugin = NativeAMPPlugin(trainer)
        self.apex_plugin = ApexPlugin(trainer)

    def on_trainer_init(self, plugins):
        self.plugins = plugins
        if self.plugins is None:
            self.plugins = []

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
            self.trainer.amp_backend = AMPType.NATIVE
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
