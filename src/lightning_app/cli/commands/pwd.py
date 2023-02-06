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
import sys

from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from lightning_app.cli.commands.cd import _CD_FILE
from lightning_app.utilities.app_helpers import Logger

logger = Logger(__name__)


def pwd() -> str:
    """Print your current working directory in the Lightning Cloud filesystem."""

    if sys.platform == "win32":
        print("`pwd` isn't supported on windows. Open an issue on Github.")
        sys.exit(0)

    with Live(Spinner("point", text=Text("pending...", style="white")), transient=True):

        root = _pwd()

    print(root)

    return root


def _pwd() -> str:
    root = "/"

    if not os.path.exists(_CD_FILE):
        with open(_CD_FILE, "w") as f:
            f.write(root + "\n")
    else:
        with open(_CD_FILE) as f:
            lines = f.readlines()
            root = lines[0].replace("\n", "")

    return root
