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

import warnings


def ignore_scalar_return_in_dp():
    # Users get confused by this warning so we silence it
    warnings.filterwarnings('ignore', message='Was asked to gather along dimension 0, but all'
                                              ' input tensors were scalars; will instead unsqueeze'
                                              ' and return a vector.')


ignore_scalar_return_in_dp()
