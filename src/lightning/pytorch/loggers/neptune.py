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
"""
Neptune Logger (Deprecated)
---------------------------

Neptune has been sunset. Please migrate to LitLogger.
"""

from typing import Any


class NeptuneLogger:
    """Deprecated: NeptuneLogger is no longer supported.

    Neptune has been sunset and this logger is no longer maintained.
    Please migrate to **LitLogger**, the recommended logging solution for
    tracking AI experiments, metrics, inputs, outputs, and artifacts.

    See: https://github.com/lightning-ai/litlogger

    Raises:
        RuntimeError: Always raised when attempting to instantiate this logger.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Raise an error indicating Neptune logger is no longer supported."""
        raise RuntimeError(
            "NeptuneLogger is no longer supported. Neptune has been sunset.\n"
            "\n"
            "Please migrate to LitLogger, the recommended logger for AI experiments.\n"
            "See: https://github.com/lightning-ai/litlogger\n"
        )
