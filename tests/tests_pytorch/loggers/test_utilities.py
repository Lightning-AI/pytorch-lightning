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

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.utilities import _version


def test_version(tmp_path):
    """Verify versions of loggers are concatenated properly."""
    logger1 = CSVLogger(tmp_path, version=0)
    logger2 = CSVLogger(tmp_path, version=2)
    logger3 = CSVLogger(tmp_path, version=1)
    logger4 = CSVLogger(tmp_path, version=0)
    loggers = [logger1, logger2, logger3, logger4]
    version = _version([])
    assert version == ""
    version = _version([logger3])
    assert version == 1
    version = _version(loggers)
    assert version == "0_2_1"
    version = _version(loggers, "-")
    assert version == "0-2-1"
