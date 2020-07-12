import os

import pytest
import torch
import tests.base.develop_pipelines as tpipes
import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate


def test_model_saves_on_cpu(tmpdir):
    """Test that ONNX model saves on CPU and size is greater than 3 MB"""
    model = EvalModelTemplate()
    trainer = Trainer(max_epochs=1)
    trainer.fit(model)

    file_path = os.path.join(tmpdir, "model.onxx")
    model.to_onnx(file_path)
    assert os.path.exists(file_path) is True
    assert os.path.getsize(file_path) > 3e+06


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_model_saves_on_multi_gpu(tmpdir):
    """Test that ONNX model saves on a distributed backend"""
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        distributed_backend='ddp_spawn',
        progress_bar_refresh_rate=0
    )

    model = EvalModelTemplate()

    tpipes.run_model_test(trainer_options, model)

    file_path = os.path.join(tmpdir, "model.onxx")
    model.to_onnx(file_path)
    assert os.path.exists(file_path) is True


def test_verbose_param(tmpdir, capsys):
    """Test that output is present when verbose parameter is set"""
    model = EvalModelTemplate()
    file_path = os.path.join(tmpdir, "model.onxx")
    model.to_onnx(file_path, verbose=True)
    captured = capsys.readouterr()
    assert "graph(%0" in captured.out


def test_input_param_with_dataloader(tmpdir):
    """Test that ONXX model is saved when a dataloader is passed in as input"""
    model = EvalModelTemplate()
    dataloader = model.dataloader(train=True)
    file_path = os.path.join(tmpdir, "model.onxx")
    model.to_onnx(file_path, input=dataloader)
    assert os.path.exists(file_path) is True


def test_input_param_with_tensor(tmpdir):
    """Test that ONXX model is saved when a a tensor is passed in as input"""
    model = EvalModelTemplate()
    tensor = torch.randn((1, 28, 28))
    file_path = os.path.join(tmpdir, "model.onxx")
    model.to_onnx(file_path, input=tensor)
    assert os.path.exists(file_path) is True
