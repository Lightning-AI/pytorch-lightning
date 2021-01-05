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

r"""
Lambda Callback
^^^^^^^^^^^^^^^

Create a simple callback on the fly using lambda functions.

"""

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class LambdaCallback(Callback):
    r"""
    Create a simple callback on the fly using lambda functions.

    Args:
        **kwargs: hooks supported by ``Callback``

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import LambdaCallback
        >>> trainer = Trainer(callbacks=[LambdaCallback(setup=lambda *args: print('setup'))])
    """

    def __init__(self, **kwargs):
        hooks = [m for m in dir(Callback) if not m.startswith("_")]
        for k, v in kwargs.items():
            if k not in hooks:
                raise MisconfigurationException(
                    f"The event function: `{k}` doesn't exist in supported callbacks function. Currently, Callback implements the following functions {dir(Callback)}"
                )
            setattr(self, k, v)
