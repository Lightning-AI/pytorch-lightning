import pathlib

from lightning.app.utilities.packaging.app_config import AppConfig, _get_config_file


def _make_empty_config_file(folder):
    file = pathlib.Path(folder / ".lightning")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()
    return file


def test_get_config_file(tmpdir):
    _ = _make_empty_config_file(tmpdir)
    config_file1 = _make_empty_config_file(tmpdir)

    assert _get_config_file(tmpdir) == pathlib.Path(tmpdir, ".lightning")
    assert _get_config_file(config_file1) == pathlib.Path(tmpdir, ".lightning")


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
