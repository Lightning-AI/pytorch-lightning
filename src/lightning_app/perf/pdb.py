import sys
import pdb

class MPPdb(pdb.Pdb):
    """
    debugger for forked programs
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def set_trace(*args, **kwargs):
    MPPdb().set_trace(*args, **kwargs)
