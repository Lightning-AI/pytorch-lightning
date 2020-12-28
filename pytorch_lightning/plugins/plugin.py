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
from pytorch_lightning.utilities import AMPType


class LightningPlugin:
    """
    Defines base class for Plugins. Plugins represent functionality that can be injected into the lightning codebase.
    """

    def required_plugins(self, amp_backend: AMPType, trainer) -> list:
        """
            Override to define additional required plugins. This is useful for when custom plugins
            need to enforce override of other plugins.

        Returns: Optional list of plugins containing additional plugins.

        Example::
            class MyPlugin(DDPPlugin):
                def required_plugins(self):
                    return [MyCustomAMPPlugin()]

            # Will automatically add the necessary AMP plugin
            trainer = Trainer(plugins=[MyPlugin()])

            # Crash as MyPlugin enforces custom AMP plugin
            trainer = Trainer(plugins=[MyPlugin(), NativeAMPPlugin()])

        """
        return []
