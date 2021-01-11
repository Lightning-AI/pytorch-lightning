
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
# limitations under the License
from functools import partial

from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.plugins.ddp_comm_hooks.allreduce_invalid import ALLREDUCE_INVALID
from pytorch_lightning.utilities import InvalidLossStrategy, LightningEnum, TORCH_GREATER_EQUAL_1_7_0
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class DDP_COMM_CALLBACK(LightningEnum):
    ON_BEFORE_BACKWARD_ENGINE_EXECUTION = "ON_BEFORE_BACKWARD_ENGINE_EXECUTION"


def initialize_ddp_comm_hooks(model: LightningDistributedDataParallel, trainer):
    if not TORCH_GREATER_EQUAL_1_7_0:
        raise MisconfigurationException(
            "Communication Hooks are introduced in PyTorch 1.7.0. Please, upgrade PyTorch to use this feature"
        )
    if model.module.invalid_loss_strategy == InvalidLossStrategy.NEVER_SKIP:
        _ddp_comm_hook_wrapper(model, trainer, LightningDDPCommHookType["ALLREDUCE_INVALID"])


def _ddp_comm_hook_wrapper(model, trainer, ddp_comm_hook):
    hook_name = ddp_comm_hook.__name__
    hook = ddp_comm_hook.HOOK
    init_state_hook = ddp_comm_hook.INIT_STATE_HOOK
    update_hook = ddp_comm_hook.ON_BEFORE_BACKWARD_ENGINE_EXECUTION
    err_msg = ddp_comm_hook.ERR_MSG

    if hook is None or init_state_hook is None:
        raise MisconfigurationException(
            "Hooks for DDP Comm Hook {hook_name} are not None. "
            f"{err_msg}"
        )
    
    trainer.add_comm_hook_state(hook_name, init_hook=init_state_hook(trainer))
    state = trainer.comm_hook_state[hook_name]
    state[DDP_COMM_CALLBACK.ON_BEFORE_BACKWARD_ENGINE_EXECUTION.value] = partial(update_hook, state=state)
    model._register_comm_hook(state, hook)


LightningDDPCommHookType = {
    "ALLREDUCE_INVALID": ALLREDUCE_INVALID
}
