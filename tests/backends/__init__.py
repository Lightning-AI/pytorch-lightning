from functools import wraps

try:
    from dtrun.launcher import DDPLauncher
except ImportError:
    class DDPLauncher:
        def run(cmd_line, **kwargs):
            def inner(func):
                @wraps(func)
                def func_wrapper(*args, **kwargs):
                    pass

                return func_wrapper
            return inner
