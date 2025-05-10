import os
from io import BytesIO
from pathlib import Path

import pytest
import torch

import tests_pytorch.helpers.pipelines as tpipes
from lightning.pytorch.demos.boring_classes import BoringModel
from tests_pytorch.helpers.runif import RunIf


@RunIf(tensorrt=True, min_cuda_gpus=1)
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


@RunIf(tensorrt=True, min_cuda_gpus=2)
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

    tpipes.run_model_test(trainer_options, model, min_acc=0.08)

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
@RunIf(tensorrt=True, min_cuda_gpus=1)
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
@RunIf(tensorrt=True, min_cuda_gpus=1)
def test_tensorrt_export_reload(output_format, ir, tmp_path):
    import torch_tensorrt

    if ir == "ts" and output_format == "exported_program":
        pytest.skip("TorchScript cannot be exported as exported_program")

    model = BoringModel()
    model.cuda().eval()
    model.example_input_array = torch.randn((4, 32))

    file_path = os.path.join(tmp_path, "model.trt")
    model.to_tensorrt(file_path, output_format=output_format, ir=ir)

    loaded_model = torch_tensorrt.load(file_path)
    if output_format == "exported_program":
        loaded_model = loaded_model.module()

    with torch.no_grad(), torch.inference_mode():
        model_output = model(model.example_input_array.to(model.device))

    jit_output = loaded_model(model.example_input_array.to("cuda"))
    assert torch.allclose(model_output, jit_output)
