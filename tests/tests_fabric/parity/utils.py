# Copyright The Lightning AI team.
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
import os

import torch


def make_deterministic():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


def get_model_input_dtype(precision):
    if precision in ("16-mixed", "16", 16):
        return torch.float16
    elif precision in ("bf16-mixed", "bf16"):
        return torch.bfloat16
    elif precision in ("64-true", "64", 64):
        return torch.double
    return torch.float32


def is_state_dict_equal(state0, state1):
    # TODO: This should be torch.equal, but MPS does not yet support this operation (torch 1.12)
    return all(torch.allclose(w0.cpu(), w1.cpu()) for w0, w1 in zip(state0.values(), state1.values()))
