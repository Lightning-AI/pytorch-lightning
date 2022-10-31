import operator
import re

import pytest

from lightning_utilities.core.imports import (
    compare_version,
    get_dependency_min_version_spec,
    module_available,
    RequirementCache,
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
    import pytest

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
    import pytest

    assert RequirementCache(f"pytest>={pytest.__version__}")
    assert not RequirementCache(f"pytest<{pytest.__version__}")
    assert "pip install -U '-'" in str(RequirementCache("-"))


def test_get_dependency_min_version_spec():
    attrs_min_version_spec = get_dependency_min_version_spec("pytest", "attrs")
    assert re.match(r"^>=[\d.]+$", attrs_min_version_spec)

    with pytest.raises(ValueError, match="'invalid' not found in package 'pytest'"):
        get_dependency_min_version_spec("pytest", "invalid")

    with pytest.raises(PackageNotFoundError, match="invalid"):
        get_dependency_min_version_spec("invalid", "invalid")
