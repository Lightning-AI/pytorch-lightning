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
from pytorch_lightning.plugins.collective.collective_plugin import Collective  # noqa: F401
from pytorch_lightning.plugins.collective.horovod_collective import HorovodCollective  # noqa: F401
from pytorch_lightning.plugins.collective.single_device_collective import SingleDeviceCollective  # noqa: F401
from pytorch_lightning.plugins.collective.torch_collective import TorchCollective  # noqa: F401
from pytorch_lightning.plugins.collective.tpu_collective import TPUCollective  # noqa: F401
