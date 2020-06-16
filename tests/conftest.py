from functools import wraps, partial
from http.server import SimpleHTTPRequestHandler

import sys
import pytest
import threading
import torch.multiprocessing as mp


def pytest_configure(config):
    config.addinivalue_line("markers", "spawn: spawn test in a separate process using torch.multiprocessing.spawn")


@pytest.mark.tryfirst
def pytest_pyfunc_call(pyfuncitem):
    if pyfuncitem.get_closest_marker("spawn"):
        testfunction = pyfuncitem.obj
        funcargs = pyfuncitem.funcargs
        testargs = tuple([funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames])

        mp.spawn(wraps, (testfunction, testargs))
        return True


@pytest.fixture
def tmpdir_server(tmpdir):
    if sys.version_info >= (3, 7):
        Handler = partial(SimpleHTTPRequestHandler, directory=str(tmpdir))
        from http.server import ThreadingHTTPServer
    else:
        # unfortunately SimpleHTTPRequestHandler doesn't accept the directory arg in python3.6
        # so we have to hack it like this
        import os

        class Handler(SimpleHTTPRequestHandler):
            def translate_path(self, path):
                # get the path from cwd
                path = super().translate_path(path)
                # get the relative path
                relpath = os.path.relpath(path, os.getcwd())
                # return the full path from root_dir
                return os.path.join(str(tmpdir), relpath)

        # ThreadingHTTPServer was added in 3.7, so we need to define it ourselves
        from socketserver import ThreadingMixIn
        from http.server import HTTPServer

        class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

    with ThreadingHTTPServer(('', 0), Handler) as server:
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        yield server.server_address
        server.shutdown()
