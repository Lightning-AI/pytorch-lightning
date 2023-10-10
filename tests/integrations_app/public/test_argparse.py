import os
import sys

from lightning.app.testing.testing import application_testing
from lightning.app.utilities.load_app import _patch_sys_argv

from integrations_app.public import _PATH_EXAMPLES


def test_app_argparse_example():
    original_argv = sys.argv

    command_line = [
        os.path.join(_PATH_EXAMPLES, "argparse", "app.py"),
        "--app_args",
        "--use_gpu",
        "--without-server",
    ]
    result = application_testing(command_line=command_line)
    assert result.exit_code == 0, result.__dict__
    assert sys.argv == original_argv


def test_patch_sys_argv():
    original_argv = sys.argv

    sys.argv = expected = ["lightning", "run", "app", "app.py"]
    with _patch_sys_argv():
        assert sys.argv == ["app.py"]

    assert sys.argv == expected

    sys.argv = expected = ["lightning", "run", "app", "app.py", "--without-server", "--env", "name=something"]
    with _patch_sys_argv():
        assert sys.argv == ["app.py"]

    assert sys.argv == expected

    sys.argv = expected = ["lightning", "run", "app", "app.py", "--app_args"]
    with _patch_sys_argv():
        assert sys.argv == ["app.py"]

    assert sys.argv == expected

    sys.argv = expected = ["lightning", "run", "app", "app.py", "--app_args", "--env", "name=something"]
    with _patch_sys_argv():
        assert sys.argv == ["app.py"]

    assert sys.argv == expected

    sys.argv = expected = [
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
        assert sys.argv == ["app.py", "--use_gpu", "--name=hello"]

    assert sys.argv == expected

    sys.argv = original_argv
