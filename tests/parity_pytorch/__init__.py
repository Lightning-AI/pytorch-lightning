import pytest

from lightning.pytorch.utilities.testing import _RunIf


def RunIf(**kwargs):
    reasons, marker_kwargs = _RunIf(**kwargs)
    return pytest.mark.skipif(condition=len(reasons) > 0, reason=f"Requires: [{' + '.join(reasons)}]", **marker_kwargs)
