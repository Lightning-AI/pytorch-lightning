from unittest import mock
from unittest.mock import call, Mock

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.torchdynamo import _TORCHDYNAMO_AVAILABLE, TorchDynamo
from pytorch_lightning.demos.boring_classes import BoringModel, ManualOptimBoringModel
from pytorch_lightning.loops import OptimizerLoop
from pytorch_lightning.loops.optimization.optimizer_loop import Closure
from pytorch_lightning.utilities.imports import _RequirementAvailable
from pytorch_lightning.utilities.model_helpers import is_overridden
from tests_pytorch.helpers.runif import RunIf


@pytest.mark.skipif(not _TORCHDYNAMO_AVAILABLE, reason=_TORCHDYNAMO_AVAILABLE.message)
def test_torchdynamo_raises():
    with pytest.raises(ValueError, match="backend 'foobar' must be"):
        TorchDynamo("foobar")
    with pytest.raises(ValueError, match="backend 'foobar' must be"):
        TorchDynamo({"train": "foobar"})
    with pytest.raises(ValueError, match="empty dictionary"):
        TorchDynamo({})

    with pytest.raises(ValueError, match=r"foobar' should be one of \['train'"):
        TorchDynamo({"foobar": "eager"})

    model = BoringModel()
    trainer = Trainer(strategy="ddp", callbacks=TorchDynamo("eager"))
    with pytest.raises(NotImplementedError, match="does not support the 'DDP"):
        trainer.fit(model)


@pytest.mark.skipif(not _TORCHDYNAMO_AVAILABLE, reason=_TORCHDYNAMO_AVAILABLE.message)
def test_torchdynamo_training_closure_cls_matches_default():
    td = TorchDynamo("eager")
    # if this fails, somebody forgot to update one
    assert td._previous_closure_cls is OptimizerLoop.closure_cls


@pytest.mark.skipif(not _TORCHDYNAMO_AVAILABLE, reason=_TORCHDYNAMO_AVAILABLE.message)
@mock.patch("torchdynamo.optimize")
@pytest.mark.parametrize("model_cls", (BoringModel, ManualOptimBoringModel))
def test_torchdynamo_mocked_context_manager(optimize_mock: Mock, model_cls):
    model = model_cls()

    compile_fn_mock = Mock()
    torchdynamo = TorchDynamo({"train": "ipex", "validate": compile_fn_mock})
    torchdynamo.default_backend = "aot_autograd"

    trainer = Trainer(
        callbacks=torchdynamo,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        num_sanity_val_steps=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(model)
    assert optimize_mock.mock_calls == [
        call("ipex"),
        call().__enter__(),
        call().__exit__(None, None, None),
        call(compile_fn_mock),
        call().__enter__(),
        call().__exit__(None, None, None),
    ]
    if model.automatic_optimization:
        assert trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.closure_cls is Closure
    else:
        assert not is_overridden("training_step", instance=model, parent=model_cls)
    if model_cls is not BoringModel:
        return

    optimize_mock.reset_mock()
    trainer.validate(model)
    assert optimize_mock.mock_calls == [call(compile_fn_mock), call().__enter__(), call().__exit__(None, None, None)]
    assert not is_overridden("validation_step", instance=model, parent=model_cls)

    optimize_mock.reset_mock()
    trainer.test(model)
    assert optimize_mock.mock_calls == [call("aot_autograd"), call().__enter__(), call().__exit__(None, None, None)]
    assert not is_overridden("test_step", instance=model, parent=model_cls)

    optimize_mock.reset_mock()
    trainer.predict(model)
    assert optimize_mock.mock_calls == [call("aot_autograd"), call().__enter__(), call().__exit__(None, None, None)]
    assert not is_overridden("predict_step", instance=model, parent=model_cls)


_NETWORKX_INSTALLED = _RequirementAvailable("networkx")


@pytest.mark.skipif(not _TORCHDYNAMO_AVAILABLE, reason=_TORCHDYNAMO_AVAILABLE.message)
@pytest.mark.parametrize("model_cls", (BoringModel, ManualOptimBoringModel))
@pytest.mark.parametrize(
    "backend",
    (
        None,
        pytest.param(
            "aot_autograd",
            marks=pytest.mark.skipif(not _NETWORKX_INSTALLED, reason=_NETWORKX_INSTALLED.message),
        ),
        pytest.param("nvfuser", marks=RunIf(min_cuda_gpus=1)),
    ),
)
def test_torchdynamo(model_cls, backend):
    model = model_cls()
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        callbacks=TorchDynamo(backend),
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
