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
Progress Bars
=============

Use or override one of the progress bar callbacks.

"""

from lightning.pytorch.callbacks.progress.progress_bar import ProgressBar  # noqa: F401
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar  # noqa: F401
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar  # noqa: F401
