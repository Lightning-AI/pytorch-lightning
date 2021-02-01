import io
import os
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import torch

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins.training_type.ddp_spawn import DDPSpawnPlugin
from pytorch_lightning.plugins.training_type.utils import on_colab_kaggle
from pytorch_lightning.utilities import _TPU_AVAILABLE, rank_zero_warn
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.seed import seed_everything

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as xla_pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    from torch_xla.core.xla_model import rendezvous
    from torch_xla.distributed.parallel_loader import ParallelLoader
else:
    xm, xla_pl, xmp, ParallelLoader, rendezvous = [None] * 5


class TPUSpawnPlugin(DDPSpawnPlugin):

    def __init__(self, parallel_devices: Sequence[int], num_nodes: int = 1, **kwargs: Dict[str, Any]) -> None:
        super().__init__(
            parallel_devices, num_nodes=num_nodes, cluster_environment=None, sync_batchnorm=False, **kwargs
        )
        self.tpu_local_core_rank = 0
        self.start_method = None

    @property
    def distributed_sampler_kwargs(self) -> dict:
        return dict(num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())

    def process_dataloader(self, dataloader: Union[Iterable, torch.utils.data.DataLoader]) -> ParallelLoader:
        device = xm.xla_device()
        dataloader = xla_pl.ParallelLoader(dataloader, [device])
        dataloader = dataloader.per_device_loader(device)
        return dataloader

    def configure_ddp(self) -> None:
        pass

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        pass

    def set_world_ranks(self, process_idx: int) -> None:
        self.tpu_local_core_rank = xm.get_local_ordinal()
        self.tpu_global_core_rank = xm.get_ordinal()
        self.global_rank = self.tpu_local_core_rank
        self.world_size = self.num_nodes * self.num_processes

    def new_process(self, process_idx: int, trainer) -> None:
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        self.set_world_ranks(process_idx)

        # set warning rank
        rank_zero_only.rank = self.global_rank

        if self.tpu_global_core_rank != 0 and trainer.progress_bar_callback is not None:
            trainer.progress_bar_callback.disable()

        self.model_to_device()
        self.barrier()

        if trainer.testing:
            results = trainer.run_test()
        else:
            results = trainer.train()

        self.__save_end_of_training_weights(self.lightning_module)
        self.transfer_distrib_spawn_state_on_fit_end(results)

    def __save_end_of_training_weights(self, model: LightningModule, trainer) -> None:
        # when training ends on these platforms dump weights to get out of the main process
        if on_colab_kaggle():
            rank_zero_warn("cleaning up... please do not interrupt")
            self.save_spawn_weights(model)

    def model_to_device(self) -> None:
        pass

    def barrier(self, name: Optional[str] = None) -> None:
        rendezvous(f"pl.Trainer.{name}")

    def on_save(self, checkpoint: dict) -> dict:
        """
        Move XLA tensors to CPU before saving
        Recommended on XLA Guide:
        https://github.com/pytorch/xla/blob/master/API_GUIDE.md#saving-and-loading-xla-tensors
        """
        return move_data_to_device(checkpoint, torch.device("cpu"))

    def broadcast(self, obj: object, src: int = 0) -> object:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        data_tensor = torch.tensor(data).to(xm.xla_device(), dtype=torch.float)
        data = xm.all_gather(data_tensor)
        buffer = io.BytesIO(data.cpu().byte().numpy())
        obj = torch.load(buffer)
        return obj

    def load_spawn_weights(self, original_model: LightningModule) -> LightningModule:
        """
        Load the temp weights saved in the process
        To recover the trained model from the ddp process we load the saved weights
        """

        loaded_model = original_model

        if self.is_global_zero:
            # load weights saved in ddp
            path = os.path.join(original_model.trainer.default_root_dir, "__temp_weight_distributed_end.ckpt")
            loaded_model = original_model.__class__.load_from_checkpoint(path)

            # copy loaded weights to old model
            original_model.load_state_dict(loaded_model.state_dict())

            # remove ddp weights
            os.remove(path)

        return loaded_model

    def save_spawn_weights(self, model: LightningModule) -> Optional[str]:
        """
        Dump a temporary checkpoint after ddp ends to get weights out of the process
        """
        if model.trainer.is_global_zero:
            path = os.path.join(model.trainer.default_root_dir, "__temp_weight_distributed_end.ckpt")
            model.trainer.save_checkpoint(path)
            return path

    def reduce_early_stopping_decision(self, should_stop: bool) -> bool:
        should_stop = torch.tensor(int(should_stop), device=self.lightning_module.device)
        stop = xm.mesh_reduce('stop_signal', should_stop, sum)
        rendezvous("pl.EarlyStoppingCallback.stop_distributed_training_check")
        should_stop = int(stop.item()) == self.world_size
        return should_stop

    def post_training(self) -> None:
        # TODO: Check if trainer references can be resolved otherwise
        model = self.lightning_module

        # restore main state with best weights
        best_path = self.mp_queue.get()
        results = self.mp_queue.get()
        last_path = self.mp_queue.get()

        # transfer back the best path to the trainer
        if self.lightning_module.trainer.checkpoint_callback is not None:
            self.lightning_module.trainer.checkpoint_callback.best_model_path = best_path
        # todo, pass also bets score

        # load last weights
        if last_path and not self.lightning_module.trainer.testing:
            ckpt = torch.load(last_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt)

        self.lightning_module = model

        # when training completes, load the weights back in main process
        self.__load_weights_on_main_process()

    def __load_weights_on_main_process(self) -> None:
        model = self.lightning_module

        # load weights if not interrupted
        # TODO: check for trainer reference
        if self.on_colab_kaggle and not model.trainer.testing:
            self.load_spawn_weights(model)

        self.lightning_module = model

    @property
    def xmp_spawn_kwargs(self):
        return {
            "args": (self.lightning_module, trainer, self.mp_queue),
            "nproc": len(self.parallel_devices),
            "start_method": self.start_method
        }

    def start_training(self, trainer) -> None:
        xmp.spawn(self.new_process, **self.xmp_spawn_kwargs)

    def start_testing(self, trainer) -> None:
        xmp.spawn(self.new_process, **self.xmp_spawn_kwargs)
