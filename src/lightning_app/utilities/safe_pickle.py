import pickle
import sys
import types
from copy import deepcopy
from pathlib import Path

from lightning_app.core.work import LightningWork
from lightning_app.utilities.app_helpers import _LightningAppRef


def get_picklable_work(work: LightningWork):
    """Pickling a LightningWork instance fails if it doesn from the work process
    itself. This function is safe to call from the work process within both MultiprocessRuntime
    and Cloud.
    Note: This function modifies the module information of the work object. Specifically, it injects
    the relative module path into the __module__ attribute of the work object. If the object is not
    importable from the CWD, then the pickle load will fail.

    Example:
        for a directory structure like below and the work class is defined in the app.py where
        the app.py is the entrypoint for the app, it will inject `foo.bar.app` into the
        __module__ attribute

        └── foo
            ├── __init__.py
            └── bar
                └── app.py
    """

    # pickling the user work class - pickling `self` will cause issue because the
    # work is running under a process, in local
    for w in _LightningAppRef.get_current().works:
        if work.name == w.name:
            # copying the work object to avoid modifying the original work object
            copied_work = deepcopy(w)
            break
    else:
        raise ValueError(f"Work with name {work.name} not found in the app references")
    # if work is defined in the __main__ or __mp__main__ (the entrypoint file for `lightning run app` command),
    # pickling/unpickling will fail, hence we need patch the module information
    if "_main__" in copied_work.__class__.__module__:
        work_class_module = sys.modules[copied_work.__class__.__module__]
        relative_path = Path(work_class_module.__file__).relative_to(Path.cwd())
        expected_module_name = relative_path.as_posix().replace(".py", "").replace("/", ".")
        # TODO @sherin: also check if the module is importable from the CWD
        fake_module = types.ModuleType(expected_module_name)
        fake_module.__dict__.update(work_class_module.__dict__)
        fake_module.__dict__["__name__"] = expected_module_name
        sys.modules[expected_module_name] = fake_module
        for k, v in fake_module.__dict__.items():
            if not k.startswith("__") and hasattr(v, "__module__"):
                if "_main__" in v.__module__:
                    v.__module__ = expected_module_name

    # removing reference to backend; backend is not picklable because of openapi client reference in it
    copied_work._backend = None
    return copied_work


def dump(work: LightningWork, f):
    picklable_work = get_picklable_work(work)
    pickle.dump(picklable_work, f)


def load(f):
    # inject current working directory to sys.path
    sys.path.insert(1, str(Path.cwd()))
    work = pickle.load(f)
    sys.path.pop(1)
    return work
