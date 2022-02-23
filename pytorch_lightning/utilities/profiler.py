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
"""Utilities for profilers."""

import os
from typing import Optional

from pytorch_lightning.utilities.cloud_io import get_filesystem


def _prepare_filename(
    profiler, action_name: Optional[str] = None, extension: str = ".txt", split_token: str = "-"
) -> str:
    args = []
    if profiler._stage is not None:
        args.append(profiler._stage)
    if profiler.filename:
        args.append(profiler.filename)
    if profiler._local_rank is not None:
        args.append(str(profiler._local_rank))
    if action_name is not None:
        args.append(action_name)
    filename = split_token.join(args) + extension
    return filename


def _prepare_streams(profiler) -> None:
    if profiler._write_stream is not None:
        return
    if profiler.filename:
        filepath = os.path.join(profiler.dirpath, _prepare_filename(profiler))
        fs = get_filesystem(filepath)
        fs.mkdirs(profiler.dirpath, exist_ok=True)
        file = fs.open(filepath, "a")
        profiler._output_file = file
        profiler._write_stream = file.write
    else:
        profiler._write_stream = profiler._rank_zero_info
