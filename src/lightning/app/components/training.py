# Copyright The Lightning AI team.
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

import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from lightning.app import structures
from lightning.app.components.python import TracerPythonScript
from lightning.app.core.flow import LightningFlow
from lightning.app.storage.path import Path
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.packaging.cloud_compute import CloudCompute

_logger = Logger(__name__)


class PyTorchLightningScriptRunner(TracerPythonScript):
    def __init__(
        self,
        script_path: str,
        script_args: Optional[Union[list, str]] = None,
        node_rank: int = 1,
        num_nodes: int = 1,
        sanity_serving: bool = False,
        cloud_compute: Optional[CloudCompute] = None,
        parallel: bool = True,
        raise_exception: bool = True,
        env: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            script_path,
            script_args,
            raise_exception=raise_exception,
            parallel=parallel,
            cloud_compute=cloud_compute,
            **kwargs,
        )
        self.node_rank = node_rank
        self.num_nodes = num_nodes
        self.best_model_path = None
        self.best_model_score = None
        self.monitor = None
        self.sanity_serving = sanity_serving
        self.has_finished = False
        self.env = env

    def configure_tracer(self):
        from lightning.pytorch import Trainer

        tracer = super().configure_tracer()
        tracer.add_traced(Trainer, "__init__", pre_fn=self._trainer_init_pre_middleware)
        return tracer

    def run(self, internal_urls: Optional[List[Tuple[str, str]]] = None, **kwargs) -> None:
        if not internal_urls:
            # Note: This is called only once.
            _logger.info(f"The node {self.node_rank} started !")
            return None

        if self.env:
            os.environ.update(self.env)

        distributed_env_vars = {
            "MASTER_ADDR": internal_urls[0][0],
            "MASTER_PORT": str(internal_urls[0][1]),
            "NODE_RANK": str(self.node_rank),
            "PL_TRAINER_NUM_NODES": str(self.num_nodes),
            "PL_TRAINER_DEVICES": "auto",
            "PL_TRAINER_ACCELERATOR": "auto",
        }

        os.environ.update(distributed_env_vars)
        return super().run(**kwargs)

    def on_after_run(self, script_globals):
        from lightning.pytorch import Trainer
        from lightning.pytorch.cli import LightningCLI

        for v in script_globals.values():
            if isinstance(v, LightningCLI):
                trainer = v.trainer
                break
            elif isinstance(v, Trainer):
                trainer = v
                break
        else:
            raise RuntimeError("No trainer instance found.")

        self.monitor = trainer.checkpoint_callback.monitor

        if trainer.checkpoint_callback.best_model_score:
            self.best_model_path = Path(trainer.checkpoint_callback.best_model_path)
            self.best_model_score = float(trainer.checkpoint_callback.best_model_score)
        else:
            self.best_model_path = Path(trainer.checkpoint_callback.last_model_path)

        self.has_finished = True

    def _trainer_init_pre_middleware(self, trainer, *args, **kwargs):
        if self.node_rank != 0:
            return {}, args, kwargs

        from lightning.pytorch.serve import ServableModuleValidator

        callbacks = kwargs.get("callbacks", [])
        if self.sanity_serving:
            callbacks = callbacks + [ServableModuleValidator()]
        kwargs["callbacks"] = callbacks
        return {}, args, kwargs

    @property
    def is_running_in_cloud(self) -> bool:
        return "LIGHTNING_APP_STATE_URL" in os.environ


class LightningTrainerScript(LightningFlow):
    def __init__(
        self,
        script_path: str,
        script_args: Optional[Union[list, str]] = None,
        num_nodes: int = 1,
        cloud_compute: CloudCompute = CloudCompute("default"),
        sanity_serving: bool = False,
        script_runner: Type[TracerPythonScript] = PyTorchLightningScriptRunner,
        **script_runner_kwargs,
    ):
        """This component enables performing distributed multi-node multi-device training.

        Example::

            from lightning.app import LightningApp
            from lightning.app.components.training import LightningTrainerScript
            from lightning.app.utilities.packaging.cloud_compute import CloudCompute

            app = LightningApp(
                LightningTrainerScript(
                    "train.py",
                    num_nodes=2,
                    cloud_compute=CloudCompute("gpu"),
                ),
            )

        Arguments:
            script_path: Path to the script to be executed.
            script_args: The arguments to be pass to the script.
            num_nodes: Number of nodes.
            cloud_compute: The cloud compute object used in the cloud.
            sanity_serving: Whether to validate that the model correctly implements
                the ServableModule API
        """
        super().__init__()
        self.script_path = script_path
        self.script_args = script_args
        self.num_nodes = num_nodes
        self.sanity_serving = sanity_serving
        self._script_runner = script_runner
        self._script_runner_kwargs = script_runner_kwargs

        self.ws = structures.List()
        for node_rank in range(self.num_nodes):
            self.ws.append(
                self._script_runner(
                    script_path=self.script_path,
                    script_args=self.script_args,
                    cloud_compute=cloud_compute,
                    node_rank=node_rank,
                    sanity_serving=self.sanity_serving,
                    num_nodes=self.num_nodes,
                    **self._script_runner_kwargs,
                )
            )

    def run(self, **run_kwargs):
        for work in self.ws:
            if all(w.internal_ip for w in self.ws):
                internal_urls = [(w.internal_ip, w.port) for w in self.ws]
                work.run(internal_urls=internal_urls, **run_kwargs)
                if all(w.has_finished for w in self.ws):
                    for w in self.ws:
                        w.stop()
            else:
                work.run()

    @property
    def best_model_score(self) -> Optional[float]:
        return self.ws[0].best_model_score

    @property
    def best_model_paths(self) -> List[Optional[Path]]:
        return [self.ws[node_idx].best_mode_path for node_idx in range(len(self.ws))]
