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

try:
    from apex import amp
except ImportError:
    amp = None


class ApexPlugin:

    def __init__(self, trainer):
        self.trainer = trainer

    def connect(self, model, optimizers=None):
        model, optimizers = self.configure_apex(model, optimizers, self.trainer.amp_level)

        if optimizers is not None:
            self.trainer.reinit_scheduler_properties(optimizers, self.trainer.lr_schedulers)
        
        return model, optimizers

    def configure_apex(self, model, optimizers, amp_level):
        if optimizers is None:
            model = amp.initialize(model, opt_level=amp_level)
        else:
            model, optimizers = amp.initialize(model, optimizers, opt_level=amp_level)

        return model, optimizers

    def training_step(self, fx, args):
        output = fx(args)
        return output
