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

from pathlib import Path

from lightning_app.storage.path import _storage_root_dir


def get_logfile(filename: str = "logs.log") -> Path:
    log_dir = Path(_storage_root_dir(), "frontend")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / filename
    return log_file
