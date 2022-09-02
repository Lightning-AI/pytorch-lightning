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
"""Test deprecated functionality which will be removed in v1.10.0."""
from unittest import mock

import numpy
import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.core.mixins.device_dtype_mixin import DeviceDtypeModuleMixin
from pytorch_lightning.demos.boring_classes import BoringModel, RandomDataset
from pytorch_lightning.overrides import LightningDistributedModule, LightningParallelModule
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.overrides.fairscale import LightningShardedDataParallel, unwrap_lightning_module_sharded
from pytorch_lightning.strategies.bagua import LightningBaguaModule
from pytorch_lightning.strategies.deepspeed import LightningDeepSpeedModule
from pytorch_lightning.strategies.ipu import LightningIPUModule
from pytorch_lightning.strategies.utils import on_colab_kaggle
from pytorch_lightning.utilities.xla_device import inner_f, pl_multi_process, XLADeviceUtils
from pytorch_lightning.utilities.apply_func import (
    apply_to_collection,
    apply_to_collections,
    convert_to_tensors,
    from_numpy,
    move_data_to_device,
    to_dtype_tensor,
    TransferableDataType,
)
from pytorch_lightning.utilities.cloud_io import atomic_save, get_filesystem, load
from pytorch_lightning.utilities.data import has_iterable_dataset, has_len
from pytorch_lightning.utilities.device_parser import (
    determine_root_gpu_device,
    is_cuda_available,
    num_cuda_devices,
    parse_cpu_cores,
    parse_gpu_ids,
    parse_tpu_cores,
)
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.utils import no_warning_call


def test_deprecated_amp_level():
    with pytest.deprecated_call(match="Setting `amp_level` inside the `Trainer` is deprecated in v1.8.0"):
        Trainer(amp_level="O3", amp_backend="apex")


@pytest.mark.parametrize(
    "wrapper_class",
    [
        LightningParallelModule,
        LightningDistributedModule,
        LightningBaguaModule,
        LightningDeepSpeedModule,
        pytest.param(LightningShardedDataParallel, marks=RunIf(fairscale=True)),
        LightningIPUModule,
    ],
)
def test_v1_10_deprecated_pl_module_init_parameter(wrapper_class):
    with no_warning_call(
        DeprecationWarning, match=rf"The argument `pl_module` in `{wrapper_class.__name__}` is deprecated in v1.8.0"
    ):
        wrapper_class(BoringModel())

    with pytest.deprecated_call(
        match=rf"The argument `pl_module` in `{wrapper_class.__name__}` is deprecated in v1.8.0"
    ):
        wrapper_class(pl_module=BoringModel())


def test_v1_10_deprecated_unwrap_lightning_module():
    with pytest.deprecated_call(match=r"The function `unwrap_lightning_module` is deprecated in v1.8.0"):
        unwrap_lightning_module(BoringModel())


@RunIf(fairscale=True)
def test_v1_10_deprecated_unwrap_lightning_module_sharded():
    with pytest.deprecated_call(match=r"The function `unwrap_lightning_module_sharded` is deprecated in v1.8.0"):
        unwrap_lightning_module_sharded(BoringModel())


def test_v1_10_deprecated_on_colab_kaggle_func():
    with pytest.deprecated_call(match="The function `on_colab_kaggle` has been deprecated in v1.8.0"):
        on_colab_kaggle()


def test_v1_10_deprecated_device_dtype_module_mixin():
    class MyModule(DeviceDtypeModuleMixin):
        pass

    with pytest.deprecated_call(match="mixins.DeviceDtypeModuleMixin` has been deprecated in v1.8.0"):
        MyModule()


def test_v1_10_deprecated_xla_device_utilities():
    with pytest.deprecated_call(match="xla_device.inner_f` has been deprecated in v1.8.0"):
        inner_f(mock.Mock(), mock.Mock())

    with pytest.deprecated_call(match="xla_device.pl_multi_process` has been deprecated in v1.8.0"):
        pl_multi_process(mock.Mock)

    with pytest.deprecated_call(match="xla_device.XLADeviceUtils` has been deprecated in v1.8.0"):
        XLADeviceUtils()

    with pytest.deprecated_call(match="xla_device.XLADeviceUtils` has been deprecated in v1.8.0"):
        XLADeviceUtils.xla_available()

    with pytest.deprecated_call(match="xla_device.XLADeviceUtils` has been deprecated in v1.8.0"):
        XLADeviceUtils.tpu_device_exists()


def test_v1_10_deprecated_apply_func_utilities():
    with pytest.deprecated_call(match="apply_func.apply_to_collection` has been deprecated in v1.8.0"):
        apply_to_collection([], dtype=object, function=(lambda x: x))

    with pytest.deprecated_call(match="apply_func.apply_to_collections` has been deprecated in v1.8.0"):
        apply_to_collections([], [], dtype=object, function=(lambda x, y: x))

    with pytest.deprecated_call(match="apply_func.convert_to_tensors` has been deprecated in v1.8.0"):
        convert_to_tensors([], torch.device("cpu"))

    with pytest.deprecated_call(match="apply_func.from_numpy` has been deprecated in v1.8.0"):
        from_numpy(numpy.zeros(2), torch.device("cpu"))

    with pytest.deprecated_call(match="apply_func.move_data_to_device` has been deprecated in v1.8.0"):
        move_data_to_device(torch.tensor(2), torch.device("cpu"))

    with pytest.deprecated_call(match="apply_func.to_dtype_tensor` has been deprecated in v1.8.0"):
        to_dtype_tensor(torch.tensor(2), dtype=torch.float32, device=torch.device("cpu"))

    class MyModule(TransferableDataType):
        pass

    with pytest.deprecated_call(match="apply_func.TransferableDataType` has been deprecated in v1.8.0"):
        MyModule()


def test_v1_10_deprecated_cloud_io_utilities(tmpdir):
    with pytest.deprecated_call(match="cloud_io.atomic_save` has been deprecated in v1.8.0"):
        atomic_save({}, tmpdir / "atomic_save.ckpt")

    with pytest.deprecated_call(match="cloud_io.get_filesystem` has been deprecated in v1.8.0"):
        get_filesystem(tmpdir)

    with pytest.deprecated_call(match="cloud_io.load` has been deprecated in v1.8.0"):
        load(str(tmpdir / "atomic_save.ckpt"))


def test_v1_10_deprecated_data_utilities():
    with pytest.deprecated_call(match="data.has_iterable_dataset` has been deprecated in v1.8.0"):
        has_iterable_dataset(DataLoader(RandomDataset(2, 4)))

    with pytest.deprecated_call(match="data.has_len` has been deprecated in v1.8.0"):
        has_len(DataLoader(RandomDataset(2, 4)))


def test_v1_10_deprecated_device_parser_utilities():
    with pytest.deprecated_call(match="device_parser.determine_root_gpu_device` has been deprecated in v1.8.0"):
        determine_root_gpu_device(None)

    with pytest.deprecated_call(match="device_parser.is_cuda_available` has been deprecated in v1.8.0"):
        is_cuda_available()

    with pytest.deprecated_call(match="device_parser.num_cuda_devices` has been deprecated in v1.8.0"):
        num_cuda_devices()

    with pytest.deprecated_call(match="device_parser.parse_cpu_cores` has been deprecated in v1.8.0"):
        parse_cpu_cores(1)

    with pytest.deprecated_call(match="device_parser.parse_gpu_ids` has been deprecated in v1.8.0"):
        parse_gpu_ids(None)

    with pytest.deprecated_call(match="device_parser.parse_tpu_cores` has been deprecated in v1.8.0"):
        parse_tpu_cores(None)
