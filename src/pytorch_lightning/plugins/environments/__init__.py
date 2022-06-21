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
from pytorch_lightning.plugins.environments.bagua_environment import BaguaEnvironment  # noqa: F401
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment  # noqa: F401
from pytorch_lightning.plugins.environments.kubeflow_environment import KubeflowEnvironment  # noqa: F401
from pytorch_lightning.plugins.environments.lightning_environment import LightningEnvironment  # noqa: F401
from pytorch_lightning.plugins.environments.lsf_environment import LSFEnvironment  # noqa: F401
from pytorch_lightning.plugins.environments.slurm_environment import SLURMEnvironment  # noqa: F401
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment  # noqa: F401
