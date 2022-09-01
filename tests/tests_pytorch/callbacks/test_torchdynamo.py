from unittest import mock
from unittest.mock import call, Mock

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.torchdynamo import _TORCHDYNAMO_AVAILABLE, _TORCHDYNAMO_CACHE, TorchDynamo
from pytorch_lightning.demos.boring_classes import BoringModel, ManualOptimBoringModel
from pytorch_lightning.loops import OptimizerLoop
from pytorch_lightning.loops.optimization.optimizer_loop import Closure
from pytorch_lightning.utilities.imports import _RequirementAvailable
from pytorch_lightning.utilities.model_helpers import is_overridden
from tests_pytorch.helpers.runif import RunIf


@pytest.mark.skipif(not _TORCHDYNAMO_AVAILABLE, reason=str(_TORCHDYNAMO_CACHE))
def test_torchdynamo_argument_parsing():
    with pytest.raises(ValueError, match="backend 'foobar' must be"):
        TorchDynamo("foobar")
    with pytest.raises(ValueError, match="backend {'fit': 'foobar'} must be"):
        TorchDynamo({"fit": "foobar"})

    td = TorchDynamo({})
    expected_backends = {
        "predict": "inductor",
        "fit": "inductor",
        "test": "inductor",
        "tune": "inductor",
        "validate": "inductor",
    }
    assert td.backends == expected_backends

    td = TorchDynamo()
    assert td.backends == expected_backends

    with pytest.raises(ValueError, match=r"foobar' should be one of \['fit'"):
        TorchDynamo({"foobar": "eager"})

    model = BoringModel()
    trainer = Trainer(strategy="ddp", callbacks=TorchDynamo("eager"))
    with pytest.raises(NotImplementedError, match="does not support the 'DDP"):
        trainer.fit(model)


@pytest.mark.skipif(not _TORCHDYNAMO_AVAILABLE, reason=str(_TORCHDYNAMO_CACHE))
def test_torchdynamo_training_closure_cls_matches_default():
    td = TorchDynamo("eager")
    # if this fails, you forgot to update one of them
    assert td._previous_closure_cls is OptimizerLoop.closure_cls


@pytest.mark.skipif(not _TORCHDYNAMO_AVAILABLE, reason=str(_TORCHDYNAMO_CACHE))
@mock.patch("torchdynamo.optimize")
@pytest.mark.parametrize("model_cls", (BoringModel, ManualOptimBoringModel))
def test_torchdynamo_mocked_context_manager(optimize_mock: Mock, tmpdir, model_cls):
    model = model_cls()

    compile_fn_mock = Mock()
    torchdynamo = TorchDynamo({"fit": "ipex", "validate": compile_fn_mock})
    default_backend = "inductor"

    trainer = Trainer(
        default_root_dir=tmpdir,
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
    assert optimize_mock.mock_calls == [call("ipex"), call().__enter__(), call().__exit__(None, None, None)] * 2
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
    assert optimize_mock.mock_calls == [call(default_backend), call().__enter__(), call().__exit__(None, None, None)]
    assert not is_overridden("test_step", instance=model, parent=model_cls)

    optimize_mock.reset_mock()
    trainer.predict(model)
    assert optimize_mock.mock_calls == [call(default_backend), call().__enter__(), call().__exit__(None, None, None)]
    assert not is_overridden("predict_step", instance=model, parent=model_cls)


_TRITON_CACHE = _RequirementAvailable("triton")


@pytest.mark.skipif(not _TORCHDYNAMO_AVAILABLE, reason=str(_TORCHDYNAMO_CACHE))
@pytest.mark.parametrize("model_cls", (BoringModel, ManualOptimBoringModel))
@pytest.mark.parametrize(
    "backend",
    (
        "aot_nop",
        "nnc",
        pytest.param("triton", marks=pytest.mark.skipif(not _TRITON_CACHE, reason=str(_TRITON_CACHE))),
        "aot_autograd",
        pytest.param("nvfuser", marks=RunIf(min_cuda_gpus=1)),
    ),
)
# `fit` runs validate already, and it should be equal to `test`, so skip it
@pytest.mark.parametrize("entrypoint", ("fit", "test", "predict"))
def test_torchdynamo(tmpdir, model_cls, backend, entrypoint):
    model = model_cls()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="auto",
        devices=1,
        callbacks=TorchDynamo(backend),
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer_fn = getattr(trainer, entrypoint)
    trainer_fn(model)
