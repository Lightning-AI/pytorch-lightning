import torch
from lightning.pytorch.utilities.signature_utils import is_param_in_hook_signature


def test_param_in_hook_signature():
    class LightningModule:
        def validation_step(self, dataloader_iter): ...

    model = LightningModule()
    assert is_param_in_hook_signature(model.validation_step, "dataloader_iter", explicit=True)

    class LightningModule:
        @torch.no_grad()
        def validation_step(self, dataloader_iter): ...

    model = LightningModule()
    assert is_param_in_hook_signature(model.validation_step, "dataloader_iter", explicit=True)

    class LightningModule:
        def validation_step(self, *args): ...

    model = LightningModule()
    assert not is_param_in_hook_signature(model.validation_step, "dataloader_iter", explicit=True)
    assert is_param_in_hook_signature(model.validation_step, "dataloader_iter", explicit=False)

    class LightningModule:
        def validation_step(self, a, b): ...

    model = LightningModule()
    assert not is_param_in_hook_signature(model.validation_step, "dataloader_iter", min_args=3)
    assert is_param_in_hook_signature(model.validation_step, "dataloader_iter", min_args=2)
