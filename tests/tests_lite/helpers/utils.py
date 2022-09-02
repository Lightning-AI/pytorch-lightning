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
import os
import numpy as np

from lightning_lite.utilities.seed import seed_everything

# generate a list of random seeds for each test
RANDOM_PORTS = list(np.random.randint(12000, 19000, 1000))


def reset_seed(seed=0):
    seed_everything(seed)


def set_random_main_port():
    reset_seed()
    port = RANDOM_PORTS.pop()
    os.environ["MASTER_PORT"] = str(port)
