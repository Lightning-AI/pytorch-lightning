from functools import wraps

try:
    from dtrun.launcher import DDPLauncher
except ImportError:
    class DDPLauncher:
        @wraps
        def run(cmd_line, **kwargs):
            def inner(func):
                pass
            return inner
