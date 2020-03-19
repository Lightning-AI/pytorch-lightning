import pytest

import torch.multiprocessing as mp


def pytest_configure(config):
    config.addinivalue_line("markers", "spawn: spawn test in a separate process using torch.multiprocessing.spawn")


def wrap(i, fn, args):
    return fn(*args)


@pytest.mark.tryfirst
def pytest_pyfunc_call(pyfuncitem):
    if pyfuncitem.get_closest_marker("spawn"):
        testfunction = pyfuncitem.obj
        funcargs = pyfuncitem.funcargs
        testargs = tuple([funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames])

        mp.spawn(wrap, (testfunction, testargs))
        return True
