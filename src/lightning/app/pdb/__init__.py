from lightning.app.pdb.pdb import MPPdb, set_trace

# Enable breakpoint within forked processes.
__builtins__["breakpoint"] = set_trace

__all__ = ["set_trace", "MPPdb"]
