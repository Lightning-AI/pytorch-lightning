import psutil


def pytest_sessionfinish(session, exitstatus):
    """Pytest hook that get called after whole test run finished, right before returning the exit status to the
    system."""
    # kill all the processes and threads created by parent
    # TODO this isn't great. We should have each tests doing it's own cleanup
    current_process = psutil.Process()
    for child in current_process.children(recursive=True):
        params = child.as_dict() or {}
        cmd_lines = params.get("cmdline", [])
        # we shouldn't kill the resource tracker from multiprocessing. If we do,
        # `atexit` will throw as it uses resource tracker to try to clean up
        if cmd_lines and "resource_tracker" in cmd_lines[-1]:
            continue
        child.kill()
