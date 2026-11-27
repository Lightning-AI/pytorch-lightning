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

    class MultiDecoratedModule:
        @torch.no_grad()
        @torch.inference_mode()
        def validation_step(self, dataloader_iter): ...

    model = MultiDecoratedModule()
    assert is_param_in_hook_signature(model.validation_step, "dataloader_iter", explicit=True)


def test_param_in_hook_signature_caches_unwrapped_function(monkeypatch):
    import inspect

    from lightning.pytorch.utilities import signature_utils

    signature_utils._get_function_argspec_info.cache_clear()

    class CachedModule:
        @torch.no_grad()
        def training_step(self, batch, batch_idx): ...

    call_count = 0
    orig_getfullargspec = inspect.getfullargspec

    def counting_getfullargspec(func):
        nonlocal call_count
        call_count += 1
        return orig_getfullargspec(func)

    monkeypatch.setattr(inspect, "getfullargspec", counting_getfullargspec)

    m1 = CachedModule()
    m2 = CachedModule()
    assert is_param_in_hook_signature(m1.training_step, "batch_idx", explicit=True)
    assert is_param_in_hook_signature(m1.training_step, "dataloader_iter", min_args=2)
    assert is_param_in_hook_signature(m2.training_step, "batch_idx", explicit=True)
    assert call_count == 1
