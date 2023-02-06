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

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lightning_app import LightningFlow


class Frontend(ABC):
    """Base class for any frontend that gets exposed by LightningFlows.

    The flow attribute will be set by the app while bootstrapping.
    """

    def __init__(self) -> None:
        self.flow: Optional["LightningFlow"] = None

    @abstractmethod
    def start_server(self, host: str, port: int, root_path: str = "") -> None:
        """Start the process that serves the UI at the given hostname and port number.

        Arguments:
            host: The hostname where the UI will be served. This gets determined by the dispatcher (e.g., cloud),
                but defaults to localhost when running locally.
            port: The port number where the UI will be served. This gets determined by the dispatcher, which by default
                chooses any free port when running locally.
            root_path: root_path for the server if app in exposed via a proxy at `/<root_path>`


        Example:

            An custom implementation could look like this:

            .. code-block:: python

                def start_server(self, host, port, root_path=""):
                    self._process = subprocess.Popen(["flask", "run" "--host", host, "--port", str(port)])
        """

    @abstractmethod
    def stop_server(self) -> None:
        """Stop the process that was started with :meth:`start_server` so the App can shut down.

        This method gets called when the LightningApp terminates.

        Example:

            .. code-block:: python

                def stop_server(self):
                    self._process.kill()
        """
