import inspect
import os
import sys
import traceback
import types
from contextlib import contextmanager
from copy import copy
from typing import Dict, List, TYPE_CHECKING, Union

from lightning_app.utilities.exceptions import MisconfigurationException

if TYPE_CHECKING:
    from lightning_app import LightningApp, LightningFlow, LightningWork

from lightning_app.utilities.app_helpers import _mock_missing_imports, Logger

logger = Logger(__name__)


def _prettifiy_exception(filepath: str):
    """Pretty print the exception that occurred when loading the app."""
    # we want to format the exception as if no frame was on top.
    exp, val, tb = sys.exc_info()
    listing = traceback.format_exception(exp, val, tb)
    # remove the entry for the first frame
    del listing[1]
    listing = [
        f"Found an exception when loading your application from {filepath}. Please, resolve it to run your app.\n\n"
    ] + listing
    logger.error("".join(listing))
    sys.exit(1)


def load_app_from_file(filepath: str, raise_exception: bool = False, mock_imports: bool = False) -> "LightningApp":
    """Load a LightningApp from a file.

    Arguments:
        filepath:  The path to the file containing the LightningApp.
        raise_exception: If True, raise an exception if the app cannot be loaded.
    """

    # Taken from StreamLit: https://github.com/streamlit/streamlit/blob/develop/lib/streamlit/script_runner.py#L313

    from lightning_app.core.app import LightningApp

    # In order for imports to work in a non-package, Python normally adds the current working directory to the
    # system path, not however when running from an entry point like the `lightning` CLI command. So we do it manually:
    sys.path.append(os.path.dirname(os.path.abspath(filepath)))

    code = _create_code(filepath)
    module = _create_fake_main_module(filepath)
    try:
        with _patch_sys_argv():
            if mock_imports:
                with _mock_missing_imports():
                    exec(code, module.__dict__)
            else:
                exec(code, module.__dict__)
    except Exception as e:
        if raise_exception:
            raise e
        _prettifiy_exception(filepath)

    apps = [v for v in module.__dict__.values() if isinstance(v, LightningApp)]
    if len(apps) > 1:
        raise MisconfigurationException(f"There should not be multiple apps instantiated within a file. Found {apps}")
    if len(apps) == 1:
        return apps[0]

    raise MisconfigurationException(
        f"The provided file {filepath} does not contain a LightningApp. Instantiate your app at the module level"
        " like so: `app = LightningApp(flow, ...)`"
    )


def _new_module(name):
    """Create a new module with the given name."""

    return types.ModuleType(name)


def open_python_file(filename):
    """Open a read-only Python file taking proper care of its encoding.

    In Python 3, we would like all files to be opened with utf-8 encoding. However, some author like to specify PEP263
    headers in their source files with their own encodings. In that case, we should respect the author's encoding.
    """
    import tokenize

    if hasattr(tokenize, "open"):  # Added in Python 3.2
        # Open file respecting PEP263 encoding. If no encoding header is
        # found, opens as utf-8.
        return tokenize.open(filename)
    else:
        return open(filename, encoding="utf-8")


def _create_code(script_path: str):
    with open_python_file(script_path) as f:
        filebody = f.read()

    return compile(
        filebody,
        # Pass in the file path so it can show up in exceptions.
        script_path,
        # We're compiling entire blocks of Python, so we need "exec"
        # mode (as opposed to "eval" or "single").
        mode="exec",
        # Don't inherit any flags or "future" statements.
        flags=0,
        dont_inherit=1,
        # Use the default optimization options.
        optimize=-1,
    )


def _create_fake_main_module(script_path):
    # Create fake module. This gives us a name global namespace to
    # execute the code in.
    module = _new_module("__main__")

    # Install the fake module as the __main__ module. This allows
    # the pickle module to work inside the user's code, since it now
    # can know the module where the pickled objects stem from.
    # IMPORTANT: This means we can't use "if __name__ == '__main__'" in
    # our code, as it will point to the wrong module!!!
    sys.modules["__main__"] = module

    # Add special variables to the module's globals dict.
    # Note: The following is a requirement for the CodeHasher to
    # work correctly. The CodeHasher is scoped to
    # files contained in the directory of __main__.__file__, which we
    # assume is the main script directory.
    module.__dict__["__file__"] = os.path.abspath(script_path)
    return module


@contextmanager
def _patch_sys_argv():
    """This function modifies the ``sys.argv`` by extracting the arguments after ``--app_args`` and removed
    everything else before executing the user app script.

    The command: ``lightning run app app.py --without-server --app_args --use_gpu --env ...`` will be converted into
    ``app.py --use_gpu``
    """
    from lightning_app.cli.lightning_cli import run_app

    original_argv = copy(sys.argv)
    # 1: Remove the CLI command
    if sys.argv[:3] == ["lightning", "run", "app"]:
        sys.argv = sys.argv[3:]

    if "--app_args" not in sys.argv:
        # 2: If app_args wasn't used, there is no arguments, so we assign the shorten arguments.
        new_argv = sys.argv[:1]
    else:
        # 3: Collect all the arguments from the CLI
        options = [p.opts[0] for p in run_app.params[1:] if p.opts[0] != "--app_args"]
        argv_slice = sys.argv
        # 4: Find the index of `app_args`
        first_index = argv_slice.index("--app_args") + 1
        # 5: Find the next argument from the CLI if any.
        matches = [
            argv_slice.index(opt) for opt in options if opt in argv_slice and argv_slice.index(opt) >= first_index
        ]
        if not matches:
            last_index = len(argv_slice)
        else:
            last_index = min(matches)
        # 6: last_index is either the fully command or the latest match from the CLI options.
        new_argv = [argv_slice[0]] + argv_slice[first_index:last_index]

    # 7: Patch the command
    sys.argv = new_argv
    yield
    # 8: Restore the command
    sys.argv = original_argv


def component_to_metadata(obj: Union["LightningWork", "LightningFlow"]) -> Dict:
    from lightning_app import LightningWork

    extras = {}

    if isinstance(obj, LightningWork):
        extras = dict(
            local_build_config=obj.local_build_config.to_dict(),
            cloud_build_config=obj.cloud_build_config.to_dict(),
            cloud_compute=obj.cloud_compute.to_dict(),
        )

    return dict(
        affiliation=obj.name.split("."),
        cls_name=obj.__class__.__name__,
        module=obj.__module__,
        docstring=inspect.getdoc(obj.__init__),
        **extras,
    )


def extract_metadata_from_app(app: "LightningApp") -> List:
    metadata = {flow.name: component_to_metadata(flow) for flow in app.flows}
    metadata.update({work.name: component_to_metadata(work) for work in app.works})
    return list(metadata[key] for key in sorted(metadata.keys()))
