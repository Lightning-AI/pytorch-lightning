import os
import sys

from lightning_app import _PROJECT_ROOT
from lightning_app.testing.helpers import RunIf
from lightning_app.utilities.tracer import Tracer


@RunIf(pytorch_lightning=True)
def test_tracer():
    from pytorch_lightning import Trainer

    def pre_fn(self, *args, **kwargs):
        kwargs["fast_dev_run"] = True
        return {}, args, kwargs

    def post_fn(self, ret):
        return {}, ret

    tracer = Tracer()
    tracer.add_traced(Trainer, "__init__", pre_fn=pre_fn, post_fn=post_fn)
    traced_file = os.path.join(_PROJECT_ROOT, "tests/core/scripts/lightning_trainer.py")
    assert os.path.exists(traced_file)
    # This is required to get the right sys.argv for `runpy``.
    sys.argv = [traced_file]
    tracer.trace(traced_file)
