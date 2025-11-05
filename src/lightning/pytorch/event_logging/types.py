from __future__ import annotations

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

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EventRecord:
    """Lightweight event container dispatched to event-logging plugins.

    Attributes:
        type: Canonical event type name (e.g., "forward", "backward", "optimizer_step", "checkpoint", "metric").
        timestamp: Event creation time (seconds since epoch, float).
        metadata: Arbitrary key-value metadata for the event. Kept minimal to avoid perf overhead.
        duration: Optional duration in seconds if applicable; may be None.
    """

    type: str
    timestamp: float
    metadata: Dict[str, Any]
    duration: Optional[float] = None
