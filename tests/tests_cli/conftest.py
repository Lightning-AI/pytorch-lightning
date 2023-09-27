import os
from threading import Thread

import pytest

os.environ["LIGHTNING_DISPATCHED"] = "1"

original_method = Thread._wait_for_tstate_lock


def fn(self, *args, timeout=None, **kwargs):
    original_method(self, *args, timeout=1, **kwargs)


Thread._wait_for_tstate_lock = fn


@pytest.fixture()
def caplog(caplog):
    """Workaround for https://github.com/pytest-dev/pytest/issues/3697.

    Setting ``filterwarnings`` with pytest breaks ``caplog`` when ``not logger.propagate``.

    """
    import logging

    root_logger = logging.getLogger()
    root_propagate = root_logger.propagate
    root_logger.propagate = True

    propagation_dict = {
        name: logging.getLogger(name).propagate
        for name in logging.root.manager.loggerDict
        if name.startswith("lightning.app")
    }
    for name in propagation_dict:
        logging.getLogger(name).propagate = True

    yield caplog

    root_logger.propagate = root_propagate
    for name, propagate in propagation_dict.items():
        logging.getLogger(name).propagate = propagate
