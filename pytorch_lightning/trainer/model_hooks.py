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

import inspect
from abc import ABC, abstractmethod

from pytorch_lightning.core.lightning import LightningModule


class TrainerModelHooksMixin(ABC):
    def is_function_implemented(self, f_name, model=None):
        if model is None:
            model = self.get_model()
        f_op = getattr(model, f_name, None)
        return callable(f_op)

    def has_arg(self, f_name, arg_name):
        model = self.get_model()
        f_op = getattr(model, f_name, None)
        return arg_name in inspect.signature(f_op).parameters

    @abstractmethod
    def get_model(self) -> LightningModule:
        """Warning: this is just empty shell for code implemented in other class."""
