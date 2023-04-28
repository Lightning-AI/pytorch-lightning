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

import multiprocessing
import os
import pdb
import sys

_stdin = [None]
_stdin_lock = multiprocessing.Lock()
try:
    _stdin_fd = sys.stdin.fileno()
except Exception:
    _stdin_fd = None


# Taken from https://github.com/facebookresearch/metaseq/blob/main/metaseq/pdb.py
class MPPdb(pdb.Pdb):
    """A Pdb wrapper that works in a multiprocessing environment."""

    def __init__(self) -> None:
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self) -> None:
        stdin_back = sys.stdin
        with _stdin_lock:
            try:
                if _stdin_fd is not None:
                    if not _stdin[0]:
                        _stdin[0] = os.fdopen(_stdin_fd)
                    sys.stdin = _stdin[0]
                self.cmdloop()
            finally:
                sys.stdin = stdin_back


def set_trace() -> None:
    pdb = MPPdb()
    pdb.set_trace(sys._getframe().f_back)
