import torch

from pytorch_lightning.plugins .training_type.ddp import DDPPlugin
from pytorch_lightning.core.step_result import Result


class DDP2Plugin(DDPPlugin):

    def setup(self, model):
        self._model = model
        # set the task idx
        self.task_idx = self.cluster_environment.local_rank()
        # the difference to DDP is that we don't call children processes here

    def reduce(self, output, *args, **kwargs):
        if isinstance(output, Result):
            output.dp_reduce()

        elif isinstance(output, torch.Tensor):
            output = output.mean()

        return output

    @property
    def root_device(self):
        return self.parallel_devices[0]

    def model_to_device(self):
        # no need to do anything when model is wrapped in torch.nn.DataParallel
        pass

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=self.num_nodes, rank=self.global_rank)
        return distributed_sampler_kwargs

    def set_world_ranks(self):
        self.local_rank = self.task_idx
        self.node_rank = self.cluster_environment.node_rank()
        self.global_rank = self.node_rank
        self.world_size = self.num_nodes
