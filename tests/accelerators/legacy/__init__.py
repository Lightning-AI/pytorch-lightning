# todo: feel free to move any of these "legacy" tests up...

try:
    from dtrun.launcher import DDPLauncher
except ImportError:

    class DDPLauncher:

        def run(cmd_line, **kwargs):

            def inner(func):
                pass

            return inner
