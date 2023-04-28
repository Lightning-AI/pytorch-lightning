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
from dataclasses import dataclass


@dataclass
class _Config:
    id = os.getenv("LIGHTNING_USER_ID")
    key = os.getenv("LIGHTNING_API_KEY")
    url = os.getenv("LIGHTNING_CLOUD_URL", "https://lightning.ai")
    api_key = os.getenv("LIGHTNING_API_KEY")
    username = os.getenv("LIGHTNING_USERNAME")
    video_location = os.getenv("VIDEO_LOCATION", "./artifacts/videos")
    har_location = os.getenv("HAR_LOCATION", "./artifacts/hars")
    slowmo = os.getenv("SLOW_MO", "0")
