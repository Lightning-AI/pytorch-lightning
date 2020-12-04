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

export PL_RUNNING_SPECIAL_TESTS=1
# Running special tests
PL_RUNNING_SPECIAL_TESTS=1 python -m coverage run --source pytorch_lightning -a -m pytest tests/plugins/test_pipe_plugin.py::test_pipe_plugin_ddp_rpc_manual --verbose --capture=no
PL_RUNNING_SPECIAL_TESTS=1 python -m coverage run --source pytorch_lightning -a -m pytest tests/plugins/test_pipe_plugin.py::test_pipe_plugin_ddp_rpc_manual_amp --verbose --capture=no
PL_RUNNING_SPECIAL_TESTS=1 python -m coverage run --source pytorch_lightning -a -m pytest tests/plugins/test_pipe_plugin.py::test_pipe_plugin_ddp_rpc_automatic --verbose --capture=no
PL_RUNNING_SPECIAL_TESTS=1 python -m coverage run --source pytorch_lightning -a -m pytest tests/plugins/test_pipe_plugin.py::test_pipe_plugin_ddp_rpc_with_wrong_balance --verbose --capture=no
