# Copyright The PyTorch Lightning team.
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
from lightning_fabric.loggers.logger import Logger  # noqa: F401
from lightning_fabric.loggers.tensorboard import TensorBoardLogger  # noqa: F401

# TODO(fabric): remove notes
# - Removed model checkpoint callback dependency and after_save_checkpoint hook
# - Removed DummyLogger
# - Moved properties to the top
# - save_dir -> root_dir


# - Removed hparams saving, and the save() method
# - Removed OmegaConf
# - save_dir -> root_dir
# - Removed log_graph argument
