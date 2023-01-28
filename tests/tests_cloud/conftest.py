import importlib
import os
import tempfile
import types
from pathlib import Path

import pytest

import lightning
import lightning.store


def reload_package(package):
    # credit: https://stackoverflow.com/a/28516918/4521646
    assert hasattr(package, "__package__")
    fn = package.__file__
    fn_dir = os.path.dirname(fn) + os.sep
    module_visit = {fn}
    del fn

    def reload_recursive_ex(module):
        importlib.reload(module)

        for module_child in vars(module).values():
            if not isinstance(module_child, types.ModuleType):
                continue
            fn_child = getattr(module_child, "__file__", None)
            if (fn_child is not None) and fn_child.startswith(fn_dir) and fn_child not in module_visit:
                # print("reloading:", fn_child, "from", module)
                module_visit.add(fn_child)
                reload_recursive_ex(module_child)

    return reload_recursive_ex(package)


@pytest.fixture(scope="function", autouse=True)
def lit_home(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dirname:
        monkeypatch.setattr(Path, "home", lambda: tmp_dirname)
        # we need to reload whole subpackage to apply the mock/fixture
        reload_package(lightning.store)
        yield os.path.join(tmp_dirname, ".lightning")
