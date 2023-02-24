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

import hashlib
from pathlib import Path


def get_hash(path: Path, chunk_num_blocks: int = 128) -> str:
    """Get the hash of a file."""
    h = hashlib.blake2b(digest_size=20)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_num_blocks * h.block_size), b""):
            h.update(chunk)
    return h.hexdigest()
