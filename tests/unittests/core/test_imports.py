import operator
import re

import pytest
from lightning_utilities.core.imports import (
    ModuleAvailableCache,
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
    assert "pip install -U '-'" in str(RequirementCache("-"))

    # invalid requirement is skipped by valid module
    assert RequirementCache(f"pytest<{pytest.__version__}", "pytest")

    cache = RequirementCache("this_module_is_not_installed")
    assert not cache
    assert "pip install -U 'this_module_is_not_installed" in str(cache)

    cache = RequirementCache("this_module_is_not_installed", "this_also_is_not")
    assert not cache
    assert "pip install -U 'this_module_is_not_installed" in str(cache)


def test_module_available_cache():
    assert ModuleAvailableCache("pytest")
    assert not ModuleAvailableCache("this_module_is_not_installed")
    assert "pip install -U this_module_is_not_installed" in str(ModuleAvailableCache("this_module_is_not_installed"))


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
