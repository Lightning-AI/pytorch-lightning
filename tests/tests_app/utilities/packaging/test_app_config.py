import pathlib

from lightning.app.utilities.packaging.app_config import _get_config_file, AppConfig


def _make_empty_config_file(folder):
    file = pathlib.Path(folder / ".lightning")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()
    return file


def test_get_config_file(tmp_path):
    _ = _make_empty_config_file(tmp_path)
    config_file1 = _make_empty_config_file(tmp_path)

    assert _get_config_file(tmp_path) == pathlib.Path(tmp_path, ".lightning")
    assert _get_config_file(config_file1) == pathlib.Path(tmp_path, ".lightning")


def test_app_config_save_load(tmp_path):
    config = AppConfig("my_app")
    config.save_to_file(tmp_path / ".lightning")
    loaded_config = AppConfig.load_from_file(tmp_path / ".lightning")
    assert config == loaded_config

    config = AppConfig("my_app2")
    config.save_to_dir(tmp_path)
    loaded_config = AppConfig.load_from_dir(tmp_path)
    assert config == loaded_config


def test_app_config_default_name():
    """Test that the default name gets auto-generated."""
    config = AppConfig()
    assert config.name
