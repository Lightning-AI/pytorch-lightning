from pathlib import Path

from lightning_utilities.cli.dependencies import (
    prune_packages_in_requirements,
    replace_oldest_version,
    replace_package_in_requirements,
)

_PATH_ROOT = Path(__file__).parent.parent.parent


def test_prune_packages(tmpdir):
    req_file = tmpdir / "requirements.txt"
    with open(req_file, "w") as fp:
        fp.writelines(["fire\n", "abc>=0.1\n"])
    prune_packages_in_requirements("abc", req_files=[str(req_file)])
    with open(req_file) as fp:
        lines = fp.readlines()
    assert lines == ["fire\n"]


def test_oldest_packages(tmpdir):
    req_file = tmpdir / "requirements.txt"
    with open(req_file, "w") as fp:
        fp.writelines(["fire>0.2\n", "abc>=0.1\n"])
    replace_oldest_version(req_files=[str(req_file)])
    with open(req_file) as fp:
        lines = fp.readlines()
    assert lines == ["fire>0.2\n", "abc==0.1\n"]


def test_replace_packages(tmpdir):
    req_file = tmpdir / "requirements.txt"
    with open(req_file, "w") as fp:
        fp.writelines(["torchvision>=0.2\n", "torch>=1.0 # comment\n", "torchtext <0.3\n"])
    replace_package_in_requirements(old_package="torch", new_package="pytorch", req_files=[str(req_file)])
    with open(req_file) as fp:
        lines = fp.readlines()
    assert lines == ["torchvision>=0.2\n", "pytorch>=1.0 # comment\n", "torchtext <0.3\n"]
