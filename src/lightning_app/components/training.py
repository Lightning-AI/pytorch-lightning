import logging
import os
from typing import List, Optional, Tuple, Union

from lightning import CloudCompute
from lightning_app import LightningFlow, structures
from lightning_app.components.python import TracerPythonScript

_logger = logging.getLogger(__name__)


class PyTorchLightningPythonScript(TracerPythonScript):
    def __init__(
        self,
        script_path: str,
        script_args: Optional[Union[list, str]] = None,
        node_rank: int = 1,
        num_nodes: int = 1,
        global_rank: int = 0,
        local_rank: int = 0,
        sanity_serving: bool = False,
        cloud_compute: Optional[CloudCompute] = None,
        **kwargs,
    ):
        super().__init__(
            script_path, script_args, raise_exception=True, parallel=True, cloud_compute=cloud_compute, **kwargs
        )
        self.node_rank = node_rank
        self.num_nodes = num_nodes
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.best_model_path: None
        self.best_model_score = None
        self.sanity_serving = sanity_serving
        self.has_finished = False
        self.master_address = None
        self.master_port = None
        self.world_size = None

    def configure_tracer(self):
        from pytorch_lightning import Trainer

        tracer = super().configure_tracer()
        # tracer.add_traced(Trainer, "__init__", pre_fn=self._trainer_init_pre_middleware)
        return tracer

    def run(self, internal_urls: Optional[List[Tuple[str, str]]] = None):
        if not internal_urls:
            _logger.info(f"The node {self.node_rank} started !")
            return

        _logger.debug(f"Internal URLS: {internal_urls}")

        self.master_address = str(internal_urls[0][0])
        self.master_port = str(internal_urls[0][1])
        devices = self.cloud_compute.devices
        self.world_size = self.num_nodes * devices

        distributed_env_vars = {
            "MASTER_ADDRESS": self.master_address,
            "MASTER_PORT": self.master_port,
            "NODE_RANK": str(self.node_rank),
            "WORLD_SIZE": str(self.world_size),
            "PL_TRAINER_NUM_NODES": str(self.num_nodes),
            "PL_TRAINER_STRATEGY": "ddp",
            "PL_TRAINER_DEVICES": str(self.cloud_compute.devices),
            "PL_TRAINER_ACCELERATOR": "auto",
        }
        _logger.info(distributed_env_vars)
        os.environ.update(distributed_env_vars)
        return super().run()

    def on_after_run(self, script_globals):
        # TODO: Why does it hang there.
        self.has_finished = True
        raise SystemExit(0)

    def _trainer_init_pre_middleware(self, trainer, *args, **kwargs):
        if self.node_rank != 0 :
            return {}, args, kwargs

        # from pytorch_lightning.serve import ServableModuleValidator

        # callbacks = kwargs.get("callbacks", [])
        # if self.sanity_serving:
        #     callbacks = callbacks + [ServableModuleValidator()]
        # kwargs["callbacks"] = callbacks
        # if self.is_running_in_cloud:
        #     kwargs["num_nodes"] = self.num_nodes
        #     kwargs["devices"] = self.cloud_compute.devices
        # else:
        #     kwargs["num_nodes"] = 1
        # kwargs["accelerator"] = "auto"
        # kwargs["plugins"] = _Environment(
        #     main_address=self.master_address,
        #     main_port=self.master_port,
        #     world_size=self.world_size,
        #     global_rank=self.global_rank,
        #     node_rank=self.node_rank,
        # )
        return {}, args, kwargs

    @property
    def is_running_in_cloud(self) -> bool:
        return "LIGHTNING_APP_STATE_URL" in os.environ


class LightningTrainingComponent(LightningFlow):
    def __init__(
        self,
        script_path: str,
        script_args: Optional[Union[list, str]] = None,
        num_nodes: int = 1,
        cloud_compute: CloudCompute = CloudCompute("cpu"),
        sanity_serving: bool = False,
    ):
        """This component enables to perform distributed training.

        Arguments:
            script_path: Path to the script to be executed.
            script_args: The arguments to be pass to the script.
            num_nodes: Number of nodes.
            cloud_compute: The cloud compute object used in the cloud.
            sanity_serving: Whether to validate the model correctly implements
                the ServableModule API
        """
        super().__init__()
        self.ws = structures.Dict()
        self.has_initialized = False
        self.script_path = script_path
        self.script_args = script_args
        self.num_nodes = num_nodes
        self._cloud_compute = cloud_compute  # TODO: Add support for cloudCOmpute
        self.sanity_serving = sanity_serving

        if not self.is_running_in_cloud and num_nodes > 1:
            _logger.info(f"This app is running locally, `num_nodes` would be mapped to devices * {num_nodes}.")

    def run(self):
        if not self.has_initialized:
            for node_rank in range(self.num_nodes):

                if self.is_running_in_cloud:
                    devices = self._cloud_compute.devices
                    global_rank = node_rank * devices if node_rank else 0
                    work_node_rank = node_rank
                    local_rank = 0
                else:
                    global_rank = node_rank
                    work_node_rank = 0
                    local_rank = node_rank

                self.ws[str(node_rank)] = PyTorchLightningPythonScript(
                    script_path=self.script_path,
                    script_args=self.script_args,
                    cloud_compute=self._cloud_compute,
                    node_rank=work_node_rank,
                    global_rank=global_rank,
                    sanity_serving=self.sanity_serving,
                    num_nodes=self.num_nodes,
                    local_rank=local_rank,
                )

            self.has_initialized = True

        for work in self.ws.values():
            if self.ready:
                internal_urls = [(w.internal_ip, w.port) for w in self.ws.values()]
                work.run(internal_urls)
                if all(w.has_finished for w in self.ws.values()):
                    self._exit("Finished training")
            else:
                work.run()

    @property
    def ready(self) -> bool:
        return all(w.internal_ip for w in self.ws.values())

    @property
    def is_running_in_cloud(self) -> bool:
        return "LIGHTNING_APP_STATE_URL" in os.environ
