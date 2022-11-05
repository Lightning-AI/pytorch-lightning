import sys
import pdb

# https://github.com/williamFalcon/forked-pdb
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def set_trace(*args, **kwargs):
    ForkedPdb().set_trace(*args, **kwargs)
