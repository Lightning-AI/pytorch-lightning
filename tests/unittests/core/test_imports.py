import operator

from lightning_utilities.core.imports import compare_version, module_available, RequirementCache


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
    assert "Requirement '-' not met" in str(RequirementCache("-"))
