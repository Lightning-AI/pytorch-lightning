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
"""General utilities"""
import platform
from distutils.version import LooseVersion
from importlib.util import find_spec

import pkg_resources
import torch


def _module_available(module_path: str) -> bool:
    """
    Check if a path is available in your environment

    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    try:
        return find_spec(module_path) is not None
    except AttributeError:
        # Python 3.6
        return False
    except ModuleNotFoundError:
        # Python 3.7+
        return False


_IS_WINDOWS = platform.system() == "Windows"

_APEX_AVAILABLE = _module_available("apex.amp")
_BOLTS_AVAILABLE = _module_available('pl_bolts')
_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and _module_available('fairscale.nn.data_parallel')
_FAIRSCALE_PIPE_AVAILABLE = (
    _FAIRSCALE_AVAILABLE and LooseVersion(torch.__version__) >= LooseVersion("1.6.0")
    and LooseVersion(pkg_resources.get_distribution('fairscale').version) <= LooseVersion("0.1.3")
)
_GROUP_AVAILABLE = not _IS_WINDOWS and _module_available('torch.distributed.group')
_HOROVOD_AVAILABLE = _module_available("horovod.torch")
_HYDRA_AVAILABLE = _module_available("hydra")
_HYDRA_EXPERIMENTAL_AVAILABLE = _module_available("hydra.experimental")
_NATIVE_AMP_AVAILABLE = _module_available("torch.cuda.amp") and hasattr(torch.cuda.amp, "autocast")
_OMEGACONF_AVAILABLE = _module_available("omegaconf")
_PYTORCH_PRUNE_AVAILABLE = _module_available('torch.nn.utils.prune')
_RPC_AVAILABLE = not _IS_WINDOWS and _module_available('torch.distributed.rpc')
_TORCHTEXT_AVAILABLE = _module_available("torchtext")
_TORCHVISION_AVAILABLE = _module_available('torchvision')
_XLA_AVAILABLE = _module_available("torch_xla")
