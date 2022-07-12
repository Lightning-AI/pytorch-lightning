import os
import sys

from lightning_app import _PACKAGE_ROOT
from lightning_app.testing.testing import application_testing
from lightning_app.utilities.load_app import _patch_sys_argv


def test_app_argparse_example():
    command_line = [
        os.path.join(os.path.dirname(os.path.dirname(_PACKAGE_ROOT)), "examples/app_argparse/app.py"),
        "--app_args",
        "--use_gpu",
        "--without-server",
    ]
    result = application_testing(command_line=command_line)
    assert result.exit_code == 0


def test_patch_sys_argv():
    original_argv = sys.argv

    sys.argv = ["lightning", "run", "app", "app.py"]
    with _patch_sys_argv():
        assert sys.argv == [sys.executable, "app.py"]

    sys.argv = ["lightning", "run", "app", "app.py", "--without-server", "--env", "name=something"]
    with _patch_sys_argv():
        assert sys.argv == [sys.executable, "app.py"]

    sys.argv = ["lightning", "run", "app", "app.py", "--app_args"]
    with _patch_sys_argv():
        assert sys.argv == [sys.executable, "app.py"]

    sys.argv = [
        "lightning",
        "run",
        "app",
        "app.py",
        "--without-server",
        "--app_args",
        "--use_gpu",
        "--name=hello",
        "--env",
        "name=something",
    ]
    with _patch_sys_argv():
        assert sys.argv == [sys.executable, "app.py", "--use_gpu", "--name=hello"]

    sys.argv = original_argv
