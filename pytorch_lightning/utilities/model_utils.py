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

from typing import Union

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule


def is_overridden(method_name: str, model: Union[LightningModule, LightningDataModule]) -> bool:
    # if you pass DataModule instead of None or a LightningModule, we use LightningDataModule as super
    # TODO - refector this function to accept model_name, instance, parent so it makes more sense
    super_object = LightningModule if not isinstance(model, LightningDataModule) else LightningDataModule

    if not hasattr(model, method_name) or not hasattr(super_object, method_name):
        # in case of calling deprecated method
        return False

    instance_attr = getattr(model, method_name)
    if not instance_attr:
        return False
    super_attr = getattr(super_object, method_name)

    # when code pointers are different, it was implemented
    if hasattr(instance_attr, 'patch_loader_code'):
        # cannot pickle __code__ so cannot verify if PatchDataloader
        # exists which shows dataloader methods have been overwritten.
        # so, we hack it by using the string representation
        is_overridden = instance_attr.patch_loader_code != str(super_attr.__code__)
    else:
        is_overridden = instance_attr.__code__ is not super_attr.__code__
    return is_overridden
