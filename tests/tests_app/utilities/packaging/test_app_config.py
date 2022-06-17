import os
import pathlib
from contextlib import contextmanager

from lightning_app.utilities.packaging.app_config import AppConfig, find_config_file


@contextmanager
def cwd(path):
    """Utility context manager for temporarily switching the current working directory (cwd)."""
    old_pwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_pwd)


def _make_empty_config_file(folder):
    file = pathlib.Path(folder / ".lightning")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()
    return file


def test_find_config_file(tmpdir):
    with cwd(pathlib.Path("/")):
        assert find_config_file() is None

    with cwd(pathlib.Path.home()):
        assert find_config_file() is None

    _ = _make_empty_config_file(tmpdir)
    config_file1 = _make_empty_config_file(tmpdir / "a" / "b")

    assert find_config_file(tmpdir) == pathlib.Path(tmpdir, ".lightning")
    assert find_config_file(config_file1) == pathlib.Path(tmpdir, "a", "b", ".lightning")
    assert find_config_file(pathlib.Path(tmpdir, "a")) == pathlib.Path(tmpdir, ".lightning")

    # the config must be a file, a folder of the same name gets ignored
    fake_config_folder = pathlib.Path(tmpdir, "fake", ".lightning")
    fake_config_folder.mkdir(parents=True)
    assert find_config_file(tmpdir) == pathlib.Path(tmpdir, ".lightning")


def test_app_config_save_load(tmpdir):
    config = AppConfig("my_app")
    config.save_to_file(tmpdir / ".lightning")
    loaded_config = AppConfig.load_from_file(tmpdir / ".lightning")
    assert config == loaded_config

    config = AppConfig("my_app2")
    config.save_to_dir(tmpdir)
    loaded_config = AppConfig.load_from_dir(tmpdir)
    assert config == loaded_config


def test_app_config_default_name():
    """Test that the default name gets auto-generated."""
    config = AppConfig()
    assert config.name
