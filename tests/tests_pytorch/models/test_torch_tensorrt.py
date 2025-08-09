import os
import re
from io import BytesIO
from pathlib import Path

import pytest
import torch

import tests_pytorch.helpers.pipelines as pipes
from lightning.pytorch.core.module import _TORCH_TRT_AVAILABLE
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import RunIf


@RunIf(max_torch="2.2.0")
def test_torch_minimum_version():
    model = BoringModel()
    with pytest.raises(
        MisconfigurationException,
        match=re.escape(f"TensorRT export requires PyTorch 2.2 or higher. Current version is {torch.__version__}."),
    ):
        model.to_tensorrt("model.trt")


@pytest.mark.skipif(_TORCH_TRT_AVAILABLE, reason="Run this test only if tensorrt is not available.")
@RunIf(min_torch="2.2.0")
def test_missing_tensorrt_package():
    model = BoringModel()
    with pytest.raises(
        ModuleNotFoundError,
        match=re.escape(f"`{type(model).__name__}.to_tensorrt` requires `torch_tensorrt` to be installed. "),
    ):
        model.to_tensorrt("model.trt")


@RunIf(tensorrt=True, min_torch="2.2.0")
def test_tensorrt_with_wrong_default_device(tmp_path):
    model = BoringModel()
    input_sample = torch.randn((1, 32))
    file_path = os.path.join(tmp_path, "model.trt")
    with pytest.raises(MisconfigurationException):
        model.to_tensorrt(file_path, input_sample, default_device="cpu")


@RunIf(tensorrt=True, min_cuda_gpus=1, min_torch="2.2.0")
def test_tensorrt_saves_with_input_sample(tmp_path):
    model = BoringModel()
    ori_device = model.device
    input_sample = torch.randn((1, 32))

    file_path = os.path.join(tmp_path, "model.trt")
    model.to_tensorrt(file_path, input_sample)

    assert os.path.isfile(file_path)
    assert os.path.getsize(file_path) > 4e2
    assert model.device == ori_device

    file_path = Path(tmp_path) / "model.trt"
    model.to_tensorrt(file_path, input_sample)
    assert os.path.isfile(file_path)
    assert os.path.getsize(file_path) > 4e2
    assert model.device == ori_device

    file_path = BytesIO()
    model.to_tensorrt(file_path, input_sample)
    assert len(file_path.getvalue()) > 4e2


@RunIf(tensorrt=True, min_cuda_gpus=1, min_torch="2.2.0")
def test_tensorrt_error_if_no_input(tmp_path):
    model = BoringModel()
    model.example_input_array = None
    file_path = os.path.join(tmp_path, "model.trt")

    with pytest.raises(
        ValueError,
        match=r"Could not export to TensorRT since neither `input_sample` nor "
        r"`model.example_input_array` attribute is set.",
    ):
        model.to_tensorrt(file_path)


@RunIf(tensorrt=True, min_cuda_gpus=2, min_torch="2.2.0")
def test_tensorrt_saves_on_multi_gpu(tmp_path):
    trainer_options = {
        "default_root_dir": tmp_path,
        "max_epochs": 1,
        "limit_train_batches": 10,
        "limit_val_batches": 10,
        "accelerator": "gpu",
        "devices": [0, 1],
        "strategy": "ddp_spawn",
        "enable_progress_bar": False,
    }

    model = BoringModel()
    model.example_input_array = torch.randn((4, 32))

    pipes.run_model_test(trainer_options, model, min_acc=0.08)

    file_path = os.path.join(tmp_path, "model.trt")
    model.to_tensorrt(file_path)

    assert os.path.exists(file_path)


@pytest.mark.parametrize(
    ("ir", "export_type"),
    [
        ("default", torch.fx.GraphModule),
        ("dynamo", torch.fx.GraphModule),
        ("ts", torch.jit.ScriptModule),
    ],
)
@RunIf(tensorrt=True, min_cuda_gpus=1, min_torch="2.2.0")
def test_tensorrt_save_ir_type(ir, export_type):
    model = BoringModel()
    model.example_input_array = torch.randn((4, 32))

    ret = model.to_tensorrt(ir=ir)
    assert isinstance(ret, export_type)


@pytest.mark.parametrize(
    "output_format",
    ["exported_program", "torchscript"],
)
@pytest.mark.parametrize(
    "ir",
    ["default", "dynamo", "ts"],
)
@RunIf(tensorrt=True, min_cuda_gpus=1, min_torch="2.2.0")
def test_tensorrt_export_reload(output_format, ir, tmp_path):
    if ir == "ts" and output_format == "exported_program":
        pytest.skip("TorchScript cannot be exported as exported_program")

    import torch_tensorrt

    model = BoringModel()
    model.cuda().eval()
    model.example_input_array = torch.ones((4, 32))

    file_path = os.path.join(tmp_path, "model.trt")
    model.to_tensorrt(file_path, output_format=output_format, ir=ir)

    loaded_model = torch_tensorrt.load(file_path)
    if output_format == "exported_program":
        loaded_model = loaded_model.module()

    with torch.no_grad(), torch.inference_mode():
        model_output = model(model.example_input_array.to("cuda"))
        jit_output = loaded_model(model.example_input_array.to("cuda"))

    assert torch.allclose(model_output, jit_output, rtol=1e-03, atol=1e-06)
