import os
import sys
from unittest import mock
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from lightning.app.cli import lightning_cli
from lightning.app.cli.cmd_pl_init import _can_encode_icon, download_frontend, pl_app


def test_pl_app_input_paths_do_not_exist(tmp_path):
    """Test that the CLI prints an error message if the code directory or the script path does not exist."""
    runner = CliRunner()

    source_dir = tmp_path / "code"
    script_file = tmp_path / "code" / "script.py"

    result = runner.invoke(lightning_cli.init_pl_app, (str(source_dir), str(script_file)))
    assert result.exit_code == 1
    assert "The given source directory does not exist:" in result.output

    source_dir.mkdir(parents=True)

    result = runner.invoke(lightning_cli.init_pl_app, (str(source_dir), str(script_file)))
    assert result.exit_code == 1
    assert "The given script path does not exist:" in result.output

    script_file_as_folder = tmp_path / "code" / "folder"
    script_file_as_folder.mkdir(parents=True)
    result = runner.invoke(lightning_cli.init_pl_app, (str(source_dir), str(script_file_as_folder)))
    assert result.exit_code == 1
    assert "The given script path must be a file, you passed:" in result.output


def test_pl_app_script_path_not_subpath(tmp_path):
    """Test that the CLI prints an error message if the provided script path is not a subpath of the source dir."""
    runner = CliRunner()

    source_dir = tmp_path / "code"
    script_file = tmp_path / "not_code" / "script.py"

    source_dir.mkdir(parents=True)
    script_file.parent.mkdir(parents=True)
    script_file.touch()

    result = runner.invoke(lightning_cli.init_pl_app, (str(source_dir), str(script_file)), catch_exceptions=False)
    assert result.exit_code == 1
    assert "The given script path must be a subpath of the source directory." in result.output


def test_pl_app_destination_app_already_exists(tmp_path, monkeypatch):
    """Test that the CLI prints an error message if an app with the same name already exists."""
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    source_dir = tmp_path / "code"
    script_file = source_dir / "script.py"
    source_dir.mkdir(parents=True)
    script_file.parent.mkdir(parents=True, exist_ok=True)
    script_file.touch()

    # monkeypatch.chdir(tmp_path)
    app_folder = tmp_path / "existing-app"
    app_folder.mkdir(parents=True)

    result = runner.invoke(lightning_cli.init_pl_app, (str(source_dir), str(script_file), "--name", "existing-app"))
    assert result.exit_code == 1
    assert "There is already an app with the name existing-app in the current working directory" in result.output


def test_pl_app_incorrect_number_of_arguments(tmp_path):
    """Test that the CLI prints an error message if more than two input arguments for the source are provided."""
    runner = CliRunner()
    result = runner.invoke(lightning_cli.init_pl_app, ("one", "two", "three"))
    assert result.exit_code == 1
    assert "Incorrect number of arguments. You passed (one, two, three) but only either one argument" in result.output


def test_pl_app_download_frontend(tmp_path):
    build_dir = tmp_path / "app" / "ui" / "build"
    download_frontend(build_dir)
    contents = os.listdir(build_dir)
    assert "index.html" in contents
    assert "static" in contents


def test_pl_app_encode_icon(monkeypatch):
    stdout_mock = Mock(wraps=sys.stdout)
    monkeypatch.setattr(sys, "stdout", stdout_mock)

    stdout_mock.encoding = "utf-8"
    assert _can_encode_icon("📂")
    assert _can_encode_icon("📄")

    stdout_mock.encoding = "ascii"
    assert not _can_encode_icon("📂")
    assert not _can_encode_icon("📄")


@pytest.mark.parametrize(
    ("cwd", "source_dir", "script_path"),
    [
        ("./", "./", "train.py"),
        ("./", "./code", "./code/train.py"),
    ],
)
@mock.patch("lightning.app.cli.cmd_pl_init.project_file_from_template")
@mock.patch("lightning.app.cli.cmd_pl_init.download_frontend")
def test_pl_app_relative_paths(_, __, cwd, source_dir, script_path, tmp_path, monkeypatch):
    source_dir = tmp_path / source_dir
    source_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_path / script_path
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.touch()
    cwd = tmp_path / cwd
    monkeypatch.chdir(cwd)

    pl_app(source_dir=str(source_dir), script_path=str(script_path), name="app-name", overwrite=False)
    assert (cwd / "app-name").is_dir()

    expected_source_files = set(os.listdir(source_dir))
    if cwd == source_dir:
        expected_source_files.remove("app-name")
    assert set(os.listdir(cwd / "app-name" / "source")) == expected_source_files
