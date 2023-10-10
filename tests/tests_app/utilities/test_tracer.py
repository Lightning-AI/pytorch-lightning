import os
import sys

from lightning.app.testing.helpers import _RunIf
from lightning.app.utilities.tracer import Tracer

from tests_app import _PROJECT_ROOT


@_RunIf(pl=True)
def test_tracer():
    from pytorch_lightning import Trainer

    def pre_fn(self, *args, **kwargs):
        kwargs["fast_dev_run"] = True
        return {}, args, kwargs

    def post_fn(self, ret):
        return {}, ret

    tracer = Tracer()
    tracer.add_traced(Trainer, "__init__", pre_fn=pre_fn, post_fn=post_fn)
    traced_file = os.path.join(_PROJECT_ROOT, "tests/tests_app/core/scripts/lightning_trainer.py")
    assert os.path.exists(traced_file)
    # This is required to get the right sys.argv for `runpy``.
    sys.argv = [traced_file]
    tracer.trace(traced_file)
