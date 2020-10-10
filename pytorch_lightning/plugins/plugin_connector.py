from pytorch_lightning.cluster_environments import ClusterEnvironment
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class PluginConnector:

    def __init__(self, trainer):
        self.trainer = trainer
        self.plugins = []
        self.cloud_environment = None

    def on_trainer_init(self, plugins):
        self.plugins = plugins
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
