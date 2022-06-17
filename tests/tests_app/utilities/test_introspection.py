from numbers import Rational

import pytest

from lightning_app import LightningApp, LightningFlow
from lightning_app.utilities.imports import _is_pytorch_lightning_available
from lightning_app.utilities.introspection import Scanner

if _is_pytorch_lightning_available():
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities.cli import LightningCLI


def test_introspection():
    """This test validates the scanner can find some class within the provided files."""
    scanner = Scanner("./tests/core/scripts/example_1.py")
    assert scanner.has_class(Rational)
    assert not scanner.has_class(LightningApp)

    scanner = Scanner("./tests/core/scripts/example_2.py")
    assert scanner.has_class(LightningApp)
    assert not scanner.has_class(LightningFlow)


@pytest.mark.skipif(not _is_pytorch_lightning_available(), reason="pytorch_lightning isn't installed.")
def test_introspection_lightning():
    """This test validates the scanner can find some PyTorch Lightning class within the provided files."""
    scanner = Scanner("./tests/core/scripts/lightning_cli.py")
    assert not scanner.has_class(Trainer)
    assert scanner.has_class(LightningCLI)

    scanner = Scanner("./tests/core/scripts/lightning_trainer.py")
    assert scanner.has_class(Trainer)
    assert not scanner.has_class(LightningCLI)


@pytest.mark.skipif(not _is_pytorch_lightning_available(), reason="pytorch_lightning isn't installed.")
def test_introspection_lightning_overrides():
    """This test validates the scanner can find all the subclasses from primitives classes from PyTorch Lightning in the
    provided files."""
    scanner = Scanner("./tests/core/scripts/lightning_cli.py")
    scanner = Scanner("./tests/core/scripts/lightning_overrides.py")
    scan = scanner.scan()
    assert sorted(scan.keys()) == [
        "Accelerator",
        "BaseProfiler",
        "Callback",
        "LightningDataModule",
        "LightningLite",
        "LightningLoggerBase",
        "LightningModule",
        "Loop",
        "Metric",
        "PrecisionPlugin",
        "Trainer",
    ]
