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
from typing import List


def _get_hash(files: List[str], algorithm: str = "blake2", chunk_num_blocks: int = 128) -> str:
    """Hashes the contents of a list of files.

    Parameters
    ----------
    files: List[Path]
        List of files.
    algorithm: str, default "blake2"
        Algorithm to hash contents. "blake2" is set by default because it
        is faster than "md5". [1]
    chunk_num_blocks: int, default 128
        Block size to user when iterating over file chunks.

    References
    ----------
    [1] https://crypto.stackexchange.com/questions/70101/blake2-vs-md5-for-checksum-file-integrity
    [2] https://stackoverflow.com/questions/1131220/get-md5-hash-of-big-files-in-python
    """
    # validate input
    if algorithm == "blake2":
        h = hashlib.blake2b(digest_size=20)
    elif algorithm == "md5":
        h = hashlib.md5()
    else:
        raise ValueError(f"Algorithm {algorithm} not supported")

    # calculate hash for all files
    for file in files:
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_num_blocks * h.block_size), b""):
                h.update(chunk)
    return h.hexdigest()
