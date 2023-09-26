import math
import os
import tarfile
from pathlib import Path

import pytest
from lightning.app.source_code.tar import MAX_SPLIT_COUNT, _get_dir_size_and_count, _get_split_size, _tar_path


def _create_files(basedir: Path):
    source_dir = basedir / "source"
    inner_dir = source_dir / "dir"
    os.makedirs(inner_dir)
    with open(source_dir / "f1", "w") as fp:
        fp.write("f1")

    with open(inner_dir / "f2", "w") as fp:
        fp.write("f2")
    return source_dir, inner_dir


def test_max_upload_parts():
    import click

    with pytest.raises(click.ClickException):
        barely_over = MAX_SPLIT_COUNT * 2**31 + 1
        _get_split_size(barely_over)


def test_almost_max_upload_parts():
    barely_under = MAX_SPLIT_COUNT * 2**31 - 1
    assert _get_split_size(barely_under) == math.ceil(barely_under / MAX_SPLIT_COUNT)


@pytest.mark.parametrize("size", [1024 * 512, 1024 * 1024 * 5])
def test_get_dir_size_and_count(tmpdir: Path, size):
    data = os.urandom(size)
    with open(os.path.join(tmpdir, "a"), "wb") as f:
        f.write(data)
    with open(os.path.join(tmpdir, "b"), "wb") as f:
        f.write(data)
    assert _get_dir_size_and_count(tmpdir, "a") == (size, 1)


def test_tar_path(tmpdir: Path, monkeypatch):
    source_dir, inner_dir = _create_files(tmpdir)

    # Test directory
    target_file = tmpdir / "target.tar.gz"
    results = _tar_path(source_path=source_dir, target_file=target_file)
    assert results.before_size > 0
    assert results.after_size > 0

    verify_dir = tmpdir / "verify"
    os.makedirs(verify_dir)
    with tarfile.open(target_file) as tar:
        tar.extractall(verify_dir)

    assert (verify_dir / "f1").exists()
    assert (verify_dir / "dir" / "f2").exists()

    # Test single file
    f2_path = inner_dir / "f2"

    target_file = tmpdir / "target_file.tar.gz"
    results = _tar_path(source_path=f2_path, target_file=target_file)
    assert results.before_size > 0
    assert results.after_size > 0

    verify_dir = tmpdir / "verify_file"
    os.makedirs(verify_dir)
    with tarfile.open(target_file) as tar:
        tar.extractall(verify_dir)

    assert (verify_dir / "f2").exists()

    # Test single file (local)
    monkeypatch.chdir(inner_dir)

    f2_path = "f2"

    target_file = tmpdir / "target_file_local.tar.gz"
    results = _tar_path(source_path=f2_path, target_file=target_file)
    assert results.before_size > 0
    assert results.after_size > 0

    verify_dir = tmpdir / "verify_file_local"
    os.makedirs(verify_dir)
    with tarfile.open(target_file) as tar:
        tar.extractall(verify_dir)

    assert (verify_dir / "f2").exists()


def test_get_split_size():
    split_size = _get_split_size(minimum_split_size=1024 * 1000 * 10, max_split_count=10000, total_size=200000000001)

    # We shouldn't go over the max split count
    assert math.ceil(200000000001 / split_size) <= 10000

    split_size = _get_split_size(
        minimum_split_size=1024 * 1000 * 10, max_split_count=10000, total_size=1024 * 500 * 1000 * 10
    )

    assert split_size == 1024 * 1000 * 10


def test_tar_path_no_compression(tmpdir):
    source_dir, _ = _create_files(tmpdir)

    target_file = tmpdir / "target.tar.gz"
    _tar_path(source_path=source_dir, target_file=target_file, compression=False)

    verify_dir = tmpdir / "verify"
    os.makedirs(verify_dir)
    with tarfile.open(target_file) as target_tar:
        target_tar.extractall(verify_dir)

    assert (verify_dir / "f1").exists()
    assert (verify_dir / "dir" / "f2").exists()
