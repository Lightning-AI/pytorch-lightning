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

TEST_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TEST_ROOT)
TEMP_PATH = os.path.join(PROJECT_ROOT, 'test_temp')

# todo: this setting `PYTHONPATH` may not be used by other evns like Conda for import packages
if PROJECT_ROOT not in os.getenv('PYTHONPATH', ""):
    splitter = ":" if os.environ.get("PYTHONPATH", "") else ""
    os.environ['PYTHONPATH'] = f'{PROJECT_ROOT}{splitter}{os.environ.get("PYTHONPATH", "")}'

# generate a list of random seeds for each test
RANDOM_PORTS = list(np.random.randint(12000, 19000, 1000))

if not os.path.isdir(TEMP_PATH):
    os.mkdir(TEMP_PATH)
