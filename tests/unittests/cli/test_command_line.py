import subprocess
import sys
from pathlib import Path

import pytest


def test_version():
    """Prints the help message for the requirements commands."""
    return_code = subprocess.call([sys.executable, "-m", "lightning_utilities.cli", "version"])  # noqa: S603
    assert return_code == 0


@pytest.mark.parametrize("args", ["positional", "optional"])
class TestRequirements:
    """Test requirements commands."""

    BASE_CMD = (sys.executable, "-m", "lightning_utilities.cli", "requirements")
    REQUIREMENTS_SAMPLE = """
# This is sample requirements file
#  with multi line comments

torchvision >=0.13.0, <0.16.0  # sample # comment
gym[classic,control] >=0.17.0, <0.27.0
ipython[all] <8.15.0  # strict
torchmetrics >=0.10.0, <1.3.0
deepspeed >=0.8.2, <=0.9.3; platform_system != "Windows"  # strict
    """

    def _create_requirements_file(self, local_path: Path, filename: str = "testing-cli-requirements.txt"):
        """Create a sample requirements file."""
        req_file = local_path / filename
        with open(req_file, "w", encoding="utf8") as fopen:
            fopen.write(self.REQUIREMENTS_SAMPLE)
        return str(req_file)

    def _build_command(self, subcommand: str, cli_params: tuple, arg_style: str):
        """Build the command for the CLI."""
        if arg_style == "positional":
            return list(self.BASE_CMD) + [subcommand] + [value for _, value in cli_params]
        if arg_style == "optional":
            return list(self.BASE_CMD) + [subcommand] + [f"--{key}={value}" for key, value in cli_params]
        raise ValueError(f"Unknown test configuration: {arg_style}")

    def test_requirements_prune_pkgs(self, args, tmp_path):
        """Prune packages from requirements files."""
        req_file = self._create_requirements_file(tmp_path)
        cli_params = (("packages", "ipython"), ("req_files", req_file))
        cmd = self._build_command("prune-pkgs", cli_params, args)
        return_code = subprocess.call(cmd)  # noqa: S603
        assert return_code == 0

    def test_requirements_set_oldest(self, args, tmp_path):
        """Set the oldest version of packages in requirement files."""
        req_file = self._create_requirements_file(tmp_path, "requirements.txt")
        cli_params = (("req_files", req_file),)
        cmd = self._build_command("set-oldest", cli_params, args)
        return_code = subprocess.call(cmd)  # noqa: S603
        assert return_code == 0

    def test_requirements_replace_pkg(self, args, tmp_path):
        """Replace a package in requirements files."""
        req_file = self._create_requirements_file(tmp_path, "requirements.txt")
        cli_params = (("old_package", "torchvision"), ("new_package", "torchtext"), ("req_files", req_file))
        cmd = self._build_command("replace-pkg", cli_params, args)
        return_code = subprocess.call(cmd)  # noqa: S603
        assert return_code == 0
