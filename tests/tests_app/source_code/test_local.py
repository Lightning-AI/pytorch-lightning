import os
import sys
import tarfile
import uuid
from pathlib import Path
from unittest import mock

import pytest
from lightning.app.source_code import LocalSourceCodeDir


def test_repository_checksum(tmp_path):
    """LocalRepository.version() generates a different version each time."""
    repository = LocalSourceCodeDir(path=Path(tmp_path))
    version_a = repository.version

    # version is different
    repository = LocalSourceCodeDir(path=Path(tmp_path))
    version_b = repository.version

    assert version_a != version_b


@pytest.mark.skipif(sys.platform == "win32", reason="this runs only on linux")
@mock.patch.dict(os.environ, {"LIGHTNING_VSCODE_WORKSPACE": "something"})
def test_local_cache_path_tmp(tmp_path):
    """LocalRepository.cache_location is under tmp."""
    repository = LocalSourceCodeDir(path=Path(tmp_path))
    assert str(repository.cache_location).startswith("/tmp")


def test_local_cache_path_home(tmp_path):
    """LocalRepository.cache_location is under home."""
    repository = LocalSourceCodeDir(path=Path(tmp_path))
    assert str(repository.cache_location).startswith(str(Path.home()))


def test_repository_package(tmp_path, monkeypatch):
    """LocalRepository.package() creates package from local dir."""
    cache_path = Path(tmp_path)
    source_path = cache_path / "nested"
    source_path.mkdir(parents=True, exist_ok=True)
    (source_path / "test.txt").write_text("test")

    repository = LocalSourceCodeDir(path=source_path)
    repository.cache_location = cache_path
    repository.package()

    # test that package is created
    for file in cache_path.glob("**/*"):
        if file.is_file() and file.name.endswith(".tar.gz"):
            assert file.name == f"{repository.version}.tar.gz"


def test_repository_lightningignore(tmp_path):
    """LocalRepository.version uses the assumed checksum correctly."""
    # write .lightningignore file
    lightningignore = """
    # ignore files in this dir
    ignore/

    """
    (tmp_path / ".lightningignore").write_text(lightningignore)
    (tmp_path / "test.txt").write_text("test")

    repository = LocalSourceCodeDir(path=Path(tmp_path))

    assert set(repository.files) == {str(tmp_path / ".lightningignore"), str(tmp_path / "test.txt")}

    # write file that needs to be ignored
    (tmp_path / "ignore").mkdir()
    (tmp_path / "ignore/test.txt").write_text(str(uuid.uuid4()))

    repository = LocalSourceCodeDir(path=Path(tmp_path))

    assert set(repository.files) == {str(tmp_path / ".lightningignore"), str(tmp_path / "test.txt")}


def test_repository_filters_with_absolute_relative_path(tmp_path):
    """.lightningignore parsing parses paths starting with / correctly."""
    lightningignore = """
    /ignore_file/test.txt

    /ignore_dir
    """
    (tmp_path / ".lightningignore").write_text(lightningignore)
    (tmp_path / "test.txt").write_text("test")

    repository = LocalSourceCodeDir(path=Path(tmp_path))

    assert set(repository.files) == {str(tmp_path / ".lightningignore"), str(tmp_path / "test.txt")}

    # write file that needs to be ignored
    (tmp_path / "ignore_file").mkdir()
    (tmp_path / "ignore_dir").mkdir()
    (tmp_path / "ignore_file/test.txt").write_text(str(uuid.uuid4()))
    (tmp_path / "ignore_dir/test.txt").write_text(str(uuid.uuid4()))

    repository = LocalSourceCodeDir(path=Path(tmp_path))

    assert set(repository.files) == {str(tmp_path / ".lightningignore"), str(tmp_path / "test.txt")}


def test_repository_lightningignore_supports_different_patterns(tmp_path):
    """.lightningignore parsing supports different patterns."""
    # write .lightningignore file
    # default github python .gitignore
    lightningignore = """
    # ignore files in this dir
    ignore/

    # Byte-compiled / optimized / DLL files
    __pycache__/
    *.py[cod]
    *$py.class

    # C extensions
    *.so

    # Distribution / packaging
    .Python
    build/
    develop-eggs/
    dist/
    downloads/
    eggs/
    .eggs/
    lib/
    lib64/
    parts/
    sdist/
    var/
    wheels/
    *.egg-info/
    .installed.cfg
    *.egg
    MANIFEST

    # PyInstaller
    #  Usually these files are written by a python script from a template
    #  before PyInstaller builds the exe, so as to inject date/other infos into it.
    *.manifest
    *.spec

    # Installer logs
    pip-log.txt
    pip-delete-this-directory.txt

    # Unit test / coverage reports
    htmlcov/
    .tox/
    .coverage
    .coverage.*
    .cache
    nosetests.xml
    coverage.xml
    *.cover
    .hypothesis/
    .pytest_cache/

    # Translations
    *.mo
    *.pot

    # Django stuff:
    *.log
    local_settings.py
    db.sqlite3

    # Flask stuff:
    instance/
    .webassets-cache

    # Scrapy stuff:
    .scrapy

    # Sphinx documentation
    docs/_build/

    # PyBuilder
    target/

    # Jupyter Notebook
    .ipynb_checkpoints

    # pyenv
    .python-version

    # celery beat schedule file
    celerybeat-schedule

    # SageMath parsed files
    *.sage.py

    # Environments
    .env
    .env.docker
    .venv
    env/
    venv/
    ENV/
    env.bak/
    venv.bak/

    # Spyder project settings
    .spyderproject
    .spyproject

    # Rope project settings
    .ropeproject

    # mkdocs documentation
    /site

    # mypy
    .mypy_cache/

    # VS Code files
    .vscode/

    # UI files
    node_modules/

    # Data files
    models/
    models/*
    !grid/openapi/models
    postgresql_data/
    redis_data/

    # Secrets folders
    secrets/

    # Built UI
    ui/

    # Ignores Grid Runner
    vendor/
    ignore_test.py

    # Ignore cov report
    *.xml

    """
    (tmp_path / ".lightningignore").write_text(lightningignore)
    (tmp_path / "test.txt").write_text("test")

    repository = LocalSourceCodeDir(path=Path(tmp_path))

    assert set(repository.files) == {str(tmp_path / ".lightningignore"), str(tmp_path / "test.txt")}

    # write file that needs to be ignored
    (tmp_path / "ignore").mkdir()
    (tmp_path / "ignore/test.txt").write_text(str(uuid.uuid4()))

    # check that version remains the same
    repository = LocalSourceCodeDir(path=Path(tmp_path))

    assert set(repository.files) == {str(tmp_path / ".lightningignore"), str(tmp_path / "test.txt")}


def test_repository_lightningignore_unpackage(tmp_path, monkeypatch):
    """.lightningignore behaves similarly to the gitignore standard."""
    lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."

    cache_path = tmp_path / "cache"
    source_path = tmp_path / "source"
    source_path.mkdir()

    # set cache location to temp dir

    lightningignore = """
    # Ignore on all levels
    *.pyc
    *__pycache__/
    build/
    .env
    # Ignore wildcard on one level
    ./*.txt
    /*.md
    ./one-level/*.txt
    /one-level/*.md
    # Ignore only relative
    ./downloads
    /relative_downloads
    # nested
    /nested//level/
    /nested/level/
    """
    (source_path / ".lightningignore").write_text(lightningignore)

    # Dir structure
    (source_path / "include.py").write_text(lorem_ipsum)
    (source_path / "exclude.pyc").write_text(lorem_ipsum)
    (source_path / "__pycache__").mkdir()
    (source_path / "__pycache__" / "exclude.py").write_text(
        lorem_ipsum
    )  # Even tho it's .py it's in excluded __pycache__ directory
    (source_path / "__pycache__" / "exclude.pyc").write_text(
        lorem_ipsum
    )  # Even tho it's .py it's in excluded __pycache__ directory
    (source_path / "build.py").write_text(lorem_ipsum)  # Common prefix with excluded build but it's not it
    (source_path / "builds").mkdir()  # Common prefix with excluded build but it's not excluded
    (source_path / "builds" / "include.py").write_text(lorem_ipsum)
    (source_path / "builds" / "__pycache__").mkdir()  # Recursively excluded
    (source_path / "builds" / "__pycache__" / "exclude.py").write_text(lorem_ipsum)
    (source_path / "build").mkdir()  # Recursively excluded
    (source_path / "build" / "exclude.db").write_text(lorem_ipsum)
    (source_path / ".env").write_text(lorem_ipsum)  # No issues with handling hidden (.dot) files
    (source_path / "downloads").mkdir()  # exclude
    (source_path / "downloads" / "something.jpeg").write_text(lorem_ipsum)
    (source_path / "relative_downloads").mkdir()  # exclude
    (source_path / "relative_downloads" / "something.jpeg").write_text(lorem_ipsum)
    (source_path / "include").mkdir()  # include
    (source_path / "include" / "exclude.pyc").write_text(lorem_ipsum)  # exclude because of *.pyc rule
    (source_path / "include" / "include.py").write_text(lorem_ipsum)  # include
    (source_path / "include" / "downloads").mkdir()  # include because it was excluded only relative to root
    (source_path / "include" / "downloads" / "something.jpeg").write_text(lorem_ipsum)
    (source_path / "include" / "relative_downloads").mkdir()  # include because it was excluded only relative to root
    (source_path / "include" / "relative_downloads" / "something.jpeg").write_text(lorem_ipsum)
    (source_path / "exclude.txt").write_text(lorem_ipsum)
    (source_path / "exclude.md").write_text(lorem_ipsum)
    (source_path / "one-level").mkdir()
    (source_path / "one-level" / "exclude.txt").write_text(lorem_ipsum)
    (source_path / "one-level" / "exclude.md").write_text(lorem_ipsum)
    (source_path / "one-level" / "include.py").write_text(lorem_ipsum)
    (source_path / "nested").mkdir()
    (source_path / "nested" / "include.py").write_text(lorem_ipsum)
    (source_path / "nested" / "level").mkdir()
    (source_path / "nested" / "level" / "exclude.py").write_text(lorem_ipsum)

    # create repo object
    repository = LocalSourceCodeDir(path=source_path)
    repository.cache_location = cache_path
    repository.package()

    unpackage_path = tmp_path / "unpackage"

    with tarfile.open(repository.package_path) as f:
        f.extractall(unpackage_path)

    assert (unpackage_path / "include.py").exists()
    assert not (unpackage_path / "exclude.pyc").exists()  # Excluded by *.pyc
    assert not (unpackage_path / "__pycache__").exists()
    assert not (
        unpackage_path / "__pycache__" / "exclude.py"
    ).exists()  # Even tho it's .py it's in excluded __pycache__ directory
    assert not (
        unpackage_path / "__pycache__" / "exclude.pyc"
    ).exists()  # Even tho it's .py it's in excluded __pycache__ directory
    assert (unpackage_path / "build.py").exists()  # Common prefix with excluded build but it's not it
    assert (unpackage_path / "builds" / "include.py").exists()
    assert not (unpackage_path / "builds" / "__pycache__").exists()  # Recursively excluded
    assert not (unpackage_path / "builds" / "__pycache__" / "exclude.py").exists()
    assert not (unpackage_path / "build").exists()  # Recursively excluded
    assert not (unpackage_path / "build" / "exclude.db").exists()
    assert not (unpackage_path / ".env").exists()  # No issues with handling hidden (.dot) files
    assert not (unpackage_path / "downloads").mkdir()  # exclude
    assert not (unpackage_path / "downloads" / "something.jpeg").exists()
    assert not (unpackage_path / "relative_downloads").mkdir()  # exclude
    assert not (unpackage_path / "relative_downloads" / "something.jpeg").exists()
    assert not (unpackage_path / "include" / "exclude.pyc").exists()  # exclude because of *.pyc rule
    assert (unpackage_path / "include" / "include.py").exists()  # include
    assert (
        unpackage_path / "include" / "downloads" / "something.jpeg"
    ).exists()  # include because it was excluded only relative to root
    assert (
        unpackage_path / "include" / "relative_downloads" / "something.jpeg"
    ).exists()  # include because it was excluded only relative to root
    assert not (unpackage_path / "exclude.txt").exists()
    assert not (unpackage_path / "exclude.md").exists()
    assert not (unpackage_path / "one-level" / "exclude.txt").exists()
    assert not (unpackage_path / "one-level" / "exclude.md").exists()
    assert (unpackage_path / "one-level" / "include.py").exists()
    assert (unpackage_path / "nested" / "include.py").exists()
    assert not (unpackage_path / "nested" / "level" / "exclude.py").exists()
