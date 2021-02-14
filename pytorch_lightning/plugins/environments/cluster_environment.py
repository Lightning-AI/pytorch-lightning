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


class ClusterEnvironment:

    def __init__(self):
        self._world_size = None

    def master_address(self):
        pass

    def master_port(self):
        pass

    def world_size(self) -> int:
        return self._world_size

    def local_rank(self) -> int:
        pass

    def node_rank(self) -> int:
        pass
