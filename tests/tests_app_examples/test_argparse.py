import os

from lightning_app import _PACKAGE_ROOT
from lightning_app.testing.testing import application_testing


def test_app_argparse_example():
    command_line = [
        os.path.join(os.path.dirname(os.path.dirname(_PACKAGE_ROOT)), "examples/app_argparse/app.py"),
        "--app_args",
        "--use_gpu",
        "--without-server",
    ]
    result = application_testing(command_line=command_line)
    assert result.exit_code == 0
