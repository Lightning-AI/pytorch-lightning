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
import platform

import pytest
import ray
import torch

import tests.base.develop_pipelines as tpipes
from tests.base import EvalModelTemplate

try:
    import horovod
    from horovod.common.util import nccl_built
except ImportError:
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True


def _nccl_available():
    if not HOROVOD_AVAILABLE:
        return False

    try:
        return nccl_built()
    except AttributeError:
        return False


@pytest.fixture
def ray_start_2_cpus():
    address_info = ray.init(num_cpus=2)
    try:
        yield address_info
    finally:
        ray.shutdown()


@pytest.fixture
def ray_start_2_gpus():
    address_info = ray.init(num_cpus=2, num_gpus=2)
    try:
        yield address_info
    finally:
        ray.shutdown()


@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
def test_horovod_cpu(tmpdir, ray_start_2_cpus):
    """Test Horovod running multi-process on CPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=10,
        num_processes=2,
        distributed_backend='horovod_ray',
        progress_bar_refresh_rate=0
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model)


@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
@pytest.mark.skipif(not _nccl_available(), reason="test requires Horovod with NCCL support")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_horovod_multi_gpu(tmpdir, ray_start_2_gpus):
    """Test Horovod with multi-GPU support."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=2,
        distributed_backend='horovod_ray',
        progress_bar_refresh_rate=0
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model)
