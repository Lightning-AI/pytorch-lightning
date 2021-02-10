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
import importlib
import platform
from distutils.version import LooseVersion

import pkg_resources
import torch


def _module_available(module_path: str) -> bool:
    """Testing if given module is avalaible in your env

    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    # todo: find a better way than try / except
    try:
        mods = module_path.split('.')
        assert mods, 'nothing given to test'
        # it has to be tested as per partets
        for i in range(len(mods)):
            module_path = '.'.join(mods[:i + 1])
            if importlib.util.find_spec(module_path) is None:
                return False
        return True
    except AttributeError:
        return False


_APEX_AVAILABLE = _module_available("apex.amp")
_NATIVE_AMP_AVAILABLE = _module_available("torch.cuda.amp") and hasattr(torch.cuda.amp, "autocast")
_OMEGACONF_AVAILABLE = _module_available("omegaconf")
_HYDRA_AVAILABLE = _module_available("hydra")
_HYDRA_EXPERIMENTAL_AVAILABLE = _module_available("hydra.experimental")
_HOROVOD_AVAILABLE = _module_available("horovod.torch")
_TORCHTEXT_AVAILABLE = _module_available("torchtext")
_XLA_AVAILABLE = _module_available("torch_xla")
_FAIRSCALE_AVAILABLE = platform.system() != 'Windows' and _module_available('fairscale.nn.data_parallel')
_RPC_AVAILABLE = platform.system() != 'Windows' and _module_available('torch.distributed.rpc')
_GROUP_AVAILABLE = platform.system() != 'Windows' and _module_available('torch.distributed.group')
_FAIRSCALE_PIPE_AVAILABLE = _FAIRSCALE_AVAILABLE and LooseVersion(
    torch.__version__
) >= LooseVersion("1.6.0") and LooseVersion(pkg_resources.get_distribution('fairscale').version
                                            ) <= LooseVersion("0.1.3")
_BOLTS_AVAILABLE = _module_available('pl_bolts')
_PYTORCH_PRUNE_AVAILABLE = _module_available('torch.nn.utils.prune')
_TORCHVISION_AVAILABLE = _module_available('torchvision')
