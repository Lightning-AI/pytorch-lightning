from functools import wraps, partial
from http.server import SimpleHTTPRequestHandler, HTTPServer

import sys
import pytest
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


def run_file_server(root_dir):
    if sys.version_info >= (3, 7):
        Handler = partial(SimpleHTTPRequestHandler, directory=root_dir)
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
                return os.path.join(root_dir, relpath)

    with HTTPServer(('', 8000), Handler) as httpd:
        httpd.serve_forever()


@pytest.fixture
def tmpdir_server(tmpdir):
    p = mp.Process(target=run_file_server, args=(str(tmpdir),))
    p.start()
    yield
    p.terminate()
