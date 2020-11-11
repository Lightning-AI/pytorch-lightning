# Copyright The PyTorch Lightning team.
# Modifications copyright Uber Technologies, Inc.
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

from pytorch_lightning.accelerators.horovod_accelerator import HorovodAccelerator

try:
    import horovod.torch as hvd
    from horovod.ray import RayExecutor
except (ModuleNotFoundError, ImportError):
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True


def get_executable_cls():
    # Only used for testing purposes, currently.
    # We need to override this in tests to ensure test path is set correctly.
    return None


class HorovodRayAccelerator(HorovodAccelerator):
    def __init__(self, trainer, cluster_environment=None):
        super().__init__(trainer, cluster_environment)
        self.nickname = 'horovod_ray'

        settings = RayExecutor.create_settings(timeout_s=30)
        self.executor = RayExecutor(settings,
                                    num_hosts=self.trainer.num_nodes,
                                    num_slots=self.trainer.num_processes,
                                    use_gpu=self.trainer.on_gpu)

    def setup(self, model):
        self.trainer.model = model
        self.executor.start(executable_cls=get_executable_cls())

    def train(self):
        results = self.executor.run(self.train_remote)
        results, state_dict, best_path = results[0]

        self.trainer.model.load_state_dict(state_dict)
        if self.trainer.checkpoint_callback:
            self.trainer.checkpoint_callback.best_model_path = best_path

        return results

    def train_remote(self):
        hvd.init()
        if self.trainer.on_gpu:
            # Horovod assigns one local GPU per process
            self.trainer.root_gpu = hvd.local_rank()

        self.setup_worker(self.trainer.model)
        results = self.train_worker()
        if hvd.rank() != 0:
            # Only want results from the first worker
            return None

        best_model_path = None
        if self.trainer.checkpoint_callback is not None:
            best_model_path = self.trainer.checkpoint_callback.best_model_path

        model = self.trainer.model
        return results, model.state_dict(), best_model_path

    def teardown(self):
        self.executor.shutdown()
