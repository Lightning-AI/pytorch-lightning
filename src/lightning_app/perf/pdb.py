import pdb
import sys
from typing import Any


class MPPdb(pdb.Pdb):
    """debugger for forked programs."""

    def interaction(self, *args: Any, **kwargs: Any) -> None:
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def set_trace(*args: Any, **kwargs: Any) -> None:
    MPPdb().set_trace(*args, **kwargs)
