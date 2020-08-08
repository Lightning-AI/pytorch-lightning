import os

import numpy as np
import pytest

from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate
from tests.base.develop_utils import reset_seed


class ModelWithManualGradTracker(EvalModelTemplate):
    def __init__(self, norm_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_grad_norms, self.norm_type = [], float(norm_type)

    # validation spoils logger's metrics with `val_loss` records
    validation_step = None
    val_dataloader = None

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # just return a loss, no log or progress bar meta
        x, y = batch
        loss_val = self.loss(y, self(x.flatten(1, -1)))
        return {'loss': loss_val}

    def on_after_backward(self):
        out, norms = {}, []
        prefix = f'grad_{self.norm_type}_norm_'
        for name, p in self.named_parameters():
            if p.grad is None:
                continue

            # `np.linalg.norm` implementation likely uses fp64 intermediates
            flat = p.grad.data.cpu().numpy().ravel()
            norm = np.linalg.norm(flat, self.norm_type)
            norms.append(norm)

            out[prefix + name] = round(norm, 3)

        # handle total norm
        norm = np.linalg.norm(norms, self.norm_type)
        out[prefix + 'total'] = round(norm, 3)
        self.stored_grad_norms.append(out)


@pytest.mark.parametrize("norm_type", [1., 1.25, 2, 3, 5, 10, 'inf'])
def test_grad_tracking(tmpdir, norm_type, rtol=5e-3):
    os.environ['PL_DEV_DEBUG'] = '1'

    # rtol=5e-3 respects the 3 decimals rounding in `.grad_norms` and above

    reset_seed()

    # use a custom grad tracking module and a list logger
    model = ModelWithManualGradTracker(norm_type)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        track_grad_norm=norm_type,
        row_log_interval=1,  # request grad_norms every batch
    )
    result = trainer.fit(model)

    assert result == 1, "Training failed"
    logged_metrics = trainer.dev_debugger.logged_metrics
    assert len(logged_metrics) == len(model.stored_grad_norms)

    # compare the logged metrics against tracked norms on `.backward`
    for mod, log in zip(model.stored_grad_norms, logged_metrics):
        common = mod.keys() & log.keys()

        log, mod = [log[k] for k in common], [mod[k] for k in common]

        assert np.allclose(log, mod, rtol=rtol)
