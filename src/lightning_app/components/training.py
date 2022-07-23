import os
from typing import List, Optional, Tuple, Union

from lightning import CloudCompute
from lightning_app import LightningFlow, structures
from lightning_app.components.python import TracerPythonScript
from lightning_app.utilities.imports import _is_pytorch_lightning_available

if _is_pytorch_lightning_available():
    from pytorch_lightning import Callback

    class IntrospectionCallback(Callback):
        def on_train_start(self, trainer, pl_module):
            print(trainer.strategy)
            print(trainer.world_size)
            print(pl_module)


class _LightningTrainerWork(TracerPythonScript):
    def __init__(
        self,
        script_path: str,
        script_args: Optional[Union[list, str]] = None,
        node_rank: int = 1,
        num_nodes: int = 1,
        global_rank: int = 0,
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
        self.best_model_path: None
        self.best_model_score = None
        self.sanity_serving = sanity_serving
        self.has_finished = False

    def configure_tracer(self):
        from pytorch_lightning import Trainer

        tracer = super().configure_tracer()
        tracer.add_traced(Trainer, "__init__", pre_fn=self._trainer_init_pre_middleware)
        return tracer

    def run(self, internal_urls: Optional[List[Tuple[str, str]]] = None):
        if not internal_urls:
            print(f"The node {self.node_rank} started !")
            return

        print(f"Internal URLS: {internal_urls}")
        master_address = str(internal_urls[0][0])
        master_port = str(internal_urls[0][1])
        devices = self.cloud_compute.devices

        distributed_env_vars = {
            "NODE_RANK": str(self.node_rank),
            "LOCAL_RANK": str(self.global_rank),
            "GLOBAL_RANK": str(self.global_rank),
            "MASTER_ADDRESS": master_address,
            "MASTER_PORT": master_port,
            "WORLD_SIZE": str(self.num_nodes * devices),
        }
        print(distributed_env_vars)
        os.environ.update(distributed_env_vars)
        return super().run()

    def on_after_run(self, script_globals):
        # TODO: Why does it hang there.
        self.has_finished = True
        raise SystemExit(0)

    def _trainer_init_pre_middleware(self, trainer, *args, **kwargs):
        from pytorch_lightning.serve import ServableModuleValidator

        callbacks = kwargs.get("callbacks", [])
        if self.sanity_serving:
            callbacks = callbacks + [ServableModuleValidator()]
        callbacks += [IntrospectionCallback()]
        kwargs["callbacks"] = callbacks
        return {}, args, kwargs


class LightningTrainingComponent(LightningFlow):
    def __init__(
        self,
        script_path: str,
        script_args: Optional[Union[list, str]] = None,
        num_nodes: int = 1,
        cloud_compute: CloudCompute = CloudCompute("cpu"),
        sanity_serving: bool = False,
    ):
        super().__init__()
        self.ws = structures.Dict()
        self.has_initialized = False
        self.script_path = script_path
        self.script_args = script_args
        self.num_nodes = num_nodes
        self._cloud_compute = cloud_compute  # TODO: Add support for cloudCOmpute
        self.sanity_serving = sanity_serving

    def run(self):
        if not self.has_initialized:
            for node_rank in range(self.num_nodes):

                if self.is_running_in_cloud:
                    devices = self._cloud_compute.devices
                    global_rank = (node_rank + 1) * devices - 1 if node_rank else 0
                    work_node_rank = node_rank
                else:
                    global_rank = node_rank
                    work_node_rank = 0

                self.ws[str(node_rank)] = _LightningTrainerWork(
                    script_path=self.script_path,
                    script_args=self.script_args,
                    cloud_compute=self._cloud_compute,
                    node_rank=work_node_rank,
                    global_rank=global_rank,
                    sanity_serving=self.sanity_serving,
                    num_nodes=self.num_nodes,
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
