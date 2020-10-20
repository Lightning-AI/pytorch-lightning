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


class PluginConnector:

    def __init__(self, trainer):
        self.trainer = trainer
        self.plugins = []
        self.cloud_environment = None

    def on_trainer_init(self, plugins):
        self.plugins = plugins
        if self.plugins is None:
            self.plugins = []

        self.__attach_cluster()

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
