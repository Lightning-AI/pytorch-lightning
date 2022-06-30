import inspect
import logging
import os
import sys
import traceback
import types
from typing import Dict, List, TYPE_CHECKING, Union

from lightning_app.utilities.exceptions import MisconfigurationException

if TYPE_CHECKING:
    from lightning_app import LightningApp, LightningFlow, LightningWork

logger = logging.getLogger(__name__)


def load_app_from_file(filepath: str) -> "LightningApp":
    # Taken from StreamLit: https://github.com/streamlit/streamlit/blob/develop/lib/streamlit/script_runner.py#L313

    from lightning_app.core.app import LightningApp

    # In order for imports to work in a non-package, Python normally adds the current working directory to the
    # system path, not however when running from an entry point like the `lightning` CLI command. So we do it manually:
    sys.path.append(os.path.dirname(os.path.abspath(filepath)))

    code = _create_code(filepath)
    module = _create_fake_main_module(filepath)
    try:
        exec(code, module.__dict__)
    except Exception:
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
