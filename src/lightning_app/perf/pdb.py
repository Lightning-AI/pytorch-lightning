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
