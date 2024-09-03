import operator
import re
from unittest import mock
from unittest.mock import Mock

import pytest

from lightning_utilities.core.imports import (
    RequirementCache,
    compare_version,
    get_dependency_min_version_spec,
    lazy_import,
    module_available,
    requires,
)

try:
    from importlib.metadata import PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import PackageNotFoundError


def test_module_exists():
    assert module_available("_pytest")
    assert module_available("_pytest.mark.structures")
    assert not module_available("_pytest.mark.asdf")
    assert not module_available("asdf")
    assert not module_available("asdf.bla.asdf")


def testcompare_version(monkeypatch):
    monkeypatch.setattr(pytest, "__version__", "1.8.9")
    assert not compare_version("pytest", operator.ge, "1.10.0")
    assert compare_version("pytest", operator.lt, "1.10.0")

    monkeypatch.setattr(pytest, "__version__", "1.10.0.dev123")
    assert compare_version("pytest", operator.ge, "1.10.0.dev123")
    assert not compare_version("pytest", operator.ge, "1.10.0.dev124")

    assert compare_version("pytest", operator.ge, "1.10.0.dev123", use_base_version=True)
    assert compare_version("pytest", operator.ge, "1.10.0.dev124", use_base_version=True)

    monkeypatch.setattr(pytest, "__version__", "1.10.0a0+0aef44c")  # dev version before rc
    assert compare_version("pytest", operator.ge, "1.10.0.rc0", use_base_version=True)
    assert not compare_version("pytest", operator.ge, "1.10.0.rc0")
    assert compare_version("pytest", operator.ge, "1.10.0", use_base_version=True)
    assert not compare_version("pytest", operator.ge, "1.10.0")


def test_requirement_cache():
    assert RequirementCache(f"pytest>={pytest.__version__}")
    assert not RequirementCache(f"pytest<{pytest.__version__}")
    assert "pip install -U 'not-found-requirement'" in str(RequirementCache("not-found-requirement"))

    # invalid requirement is skipped by valid module
    assert RequirementCache(f"pytest<{pytest.__version__}", "pytest")

    cache = RequirementCache("this_module_is_not_installed")
    assert not cache
    assert "pip install -U 'this_module_is_not_installed" in str(cache)

    cache = RequirementCache("this_module_is_not_installed", "this_also_is_not")
    assert not cache
    assert "pip install -U 'this_module_is_not_installed" in str(cache)

    cache = RequirementCache("pytest[not-valid-extra]")
    assert not cache
    assert "pip install -U 'pytest[not-valid-extra]" in str(cache)


@mock.patch("lightning_utilities.core.imports.Requirement")
@mock.patch("lightning_utilities.core.imports._version")
@mock.patch("lightning_utilities.core.imports.distribution")
def test_requirement_cache_with_extras(distribution_mock, version_mock, requirement_mock):
    requirement_mock().specifier.contains.return_value = True
    requirement_mock().name = "jsonargparse"
    requirement_mock().extras = []
    version_mock.return_value = "1.0.0"
    assert RequirementCache("jsonargparse>=1.0.0")

    with mock.patch("lightning_utilities.core.imports.RequirementCache._get_extra_requirements") as get_extra_req_mock:
        get_extra_req_mock.return_value = [
            # Extra packages, all versions satisfied
            Mock(name="extra_package1", specifier=Mock(contains=Mock(return_value=True))),
            Mock(name="extra_package2", specifier=Mock(contains=Mock(return_value=True))),
        ]
        distribution_mock.return_value = Mock(version="0.10.0")
        requirement_mock().extras = ["signatures"]
        assert RequirementCache("jsonargparse[signatures]>=1.0.0")

    with mock.patch("lightning_utilities.core.imports.RequirementCache._get_extra_requirements") as get_extra_req_mock:
        get_extra_req_mock.return_value = [
            # Extra packages, but not all versions are satisfied
            Mock(name="extra_package1", specifier=Mock(contains=Mock(return_value=True))),
            Mock(name="extra_package2", specifier=Mock(contains=Mock(return_value=False))),
        ]
        distribution_mock.return_value = Mock(version="0.10.0")
        requirement_mock().extras = ["signatures"]
        assert not RequirementCache("jsonargparse[signatures]>=1.0.0")


@mock.patch("lightning_utilities.core.imports._version")
def test_requirement_cache_with_prerelease_package(version_mock):
    version_mock.return_value = "0.11.0"
    assert RequirementCache("transformer-engine>=0.11.0")
    version_mock.return_value = "0.11.0.dev0+931b44f"
    assert not RequirementCache("transformer-engine>=0.11.0")
    version_mock.return_value = "1.10.0.dev0+931b44f"
    assert RequirementCache("transformer-engine>=0.11.0")


def test_module_available_cache():
    assert RequirementCache(module="pytest")
    assert not RequirementCache(module="this_module_is_not_installed")
    assert "pip install -U this_module_is_not_installed" in str(RequirementCache(module="this_module_is_not_installed"))


def test_get_dependency_min_version_spec():
    attrs_min_version_spec = get_dependency_min_version_spec("pytest", "attrs")
    assert re.match(r"^>=[\d.]+$", attrs_min_version_spec)

    with pytest.raises(ValueError, match="'invalid' not found in package 'pytest'"):
        get_dependency_min_version_spec("pytest", "invalid")

    with pytest.raises(PackageNotFoundError, match="invalid"):
        get_dependency_min_version_spec("invalid", "invalid")


def test_lazy_import():
    def callback_fcn():
        raise ValueError

    math = lazy_import("math", callback=callback_fcn)
    with pytest.raises(ValueError, match=""):  # noqa: PT011
        math.floor(5.1)

    module = lazy_import("asdf")
    with pytest.raises(ModuleNotFoundError, match="No module named 'asdf'"):
        print(module)

    os = lazy_import("os")
    assert os.getcwd()


@requires("torch.unknown.subpackage")
def my_torch_func(i: int) -> int:
    import torch  # noqa

    return i


def test_torch_func_raised():
    with pytest.raises(
        ModuleNotFoundError,
        match="Required dependencies not available: \nModule not found: 'torch.unknown.subpackage'. ",
    ):
        my_torch_func(42)


@requires("random")
def my_random_func(nb: int) -> int:
    from random import randint

    return randint(0, nb)


def test_rand_func_passed():
    assert 0 <= my_random_func(42) <= 42


class MyTorchClass:
    @requires("torch>99.0", "random")
    def __init__(self):
        from random import randint

        import torch  # noqa

        self._rnd = randint(1, 9)


def test_torch_class_raised():
    with pytest.raises(
        ModuleNotFoundError, match="Required dependencies not available: \nModule not found: 'torch>99.0'."
    ):
        MyTorchClass()


class MyRandClass:
    @requires("random")
    def __init__(self, nb: int):
        from random import randint

        self._rnd = randint(1, nb)


def test_rand_class_passed():
    cls = MyRandClass(42)
    assert 0 <= cls._rnd <= 42
