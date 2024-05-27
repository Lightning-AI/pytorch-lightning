import os
import json

from lightning.fabric.plugins.environments.lightning import LightningEnvironment


class VertexAICluster(LightningEnvironment):
    """
    Configures distributed training on a vertex ai custom training job,
    More information:
    https://cloud.google.com/vertex-ai/docs/training/distributed-training#cluster-spec-format
    
    Example:
        Consider a cluster with 3 nodes, each composed of 2 gpus

        The "cluster" key in CLUSTER_SPEC will be:
            {
                'workerpool0': ['cmle-training-workerpool0-d604929a6a-0:2222'],
                'workerpool1': [
                                'cmle-training-workerpool1-d604929a6a-0:2222',
                                'cmle-training-workerpool1-d604929a6a-1:2222'
                              ]
            }

        and each process scheduled will be under the "task" key, following the same example
        the three tasks will look like this:
            task0 ("chief" spawn process) -> node 0:
            {'type': 'workerpool0', 'index': 0}
            task 1 (on first node on workerpool1) -> node 1:
            {'type': 'workerpool1', 'index': 0}
            task 2 (on second node on workerpool1) -> node 2:
            {'type': 'workerpool1', 'index': 1}
    """

    def __init__(self):
        super().__init__()
        assert "CLUSTER_SPEC" in os.environ
          self.cluster_spec = json.loads(os.environ["CLUSTER_SPEC"])

    @property
    def main_address(self) -> str:
        return self.cluster_spec["cluster"]["workerpool0"][0].split(':')[0]

    @property
    def main_port(self) -> int:
        """Set common fixed MASTER_PORT port across processes
        """
        return int(self.cluster_spec["cluster"]["workerpool0"][0].split(':')[1])

    def node_rank(self) -> int:
        task = self.cluster_spec["task"]
        if task["type"] == "workerpool0":
            return 0
        else:
            return task["index"] + 1
