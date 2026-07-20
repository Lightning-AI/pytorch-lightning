# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import operator
import os
import re
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnxruntime
import pytest
import torch
from lightning_utilities import compare_version

import tests_pytorch.helpers.pipelines as tpipes
from lightning.pytorch import Trainer
from lightning.pytorch.core.module import _ONNXSCRIPT_AVAILABLE
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities.imports import _TORCH_GREATER_EQUAL_2_6
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.utilities.test_model_summary import UnorderedModel


@RunIf(onnx=True)
def test_model_saves_with_input_sample(tmp_path):
    """Test that ONNX model saves with input sample and size is greater than 3 MB."""
    model = BoringModel()
    input_sample = torch.randn((1, 32))

    file_path = os.path.join(tmp_path, "os.path.onnx")
    model.to_onnx(file_path, input_sample)
    assert os.path.isfile(file_path)
    assert os.path.getsize(file_path) > 4e2

    file_path = Path(tmp_path) / "pathlib.onnx"
    model.to_onnx(file_path, input_sample)
    assert os.path.isfile(file_path)
    assert os.path.getsize(file_path) > 4e2

    file_path = BytesIO()
    model.to_onnx(file_path=file_path, input_sample=input_sample)
    assert len(file_path.getvalue()) > 4e2


@pytest.mark.parametrize(
    "accelerator", [pytest.param("mps", marks=RunIf(mps=True)), pytest.param("gpu", marks=RunIf(min_cuda_gpus=True))]
)
@RunIf(onnx=True)
def test_model_saves_on_gpu(tmp_path, accelerator):
    """Test that model saves on gpu."""
    model = BoringModel()
    trainer = Trainer(accelerator=accelerator, devices=1, fast_dev_run=True)
    trainer.fit(model)

    file_path = os.path.join(tmp_path, "model.onnx")
    input_sample = torch.randn((1, 32))
    model.to_onnx(file_path, input_sample)
    assert os.path.isfile(file_path)
    assert os.path.getsize(file_path) > 4e2


@pytest.mark.parametrize(
    ("modelclass", "input_sample"),
    [
        (BoringModel, torch.randn(1, 32)),
        (UnorderedModel, (torch.rand(2, 3), torch.rand(2, 10))),
    ],
)
@RunIf(onnx=True)
def test_model_saves_with_example_input_array(tmp_path, modelclass, input_sample):
    """Test that ONNX model saves with example_input_array and size is greater than 3 MB."""
    model = modelclass()
    model.example_input_array = input_sample

    file_path = os.path.join(tmp_path, "model.onnx")
    model.to_onnx(file_path)
    assert os.path.exists(file_path) is True
    assert os.path.getsize(file_path) > 4e2


@RunIf(min_cuda_gpus=2, onnx=True)
def test_model_saves_on_multi_gpu(tmp_path):
    """Test that ONNX model saves on a distributed backend."""
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
    model.example_input_array = torch.randn(5, 32)

    tpipes.run_model_test(trainer_options, model, min_acc=0.08)

    file_path = os.path.join(tmp_path, "model.onnx")
    model.to_onnx(file_path)
    assert os.path.exists(file_path) is True


# todo: investigate where the logging happening in torch.onnx for PT 2.6+
@RunIf(onnx=True, max_torch="2.6.0")
def test_verbose_param(tmp_path, capsys):
    """Test that output is present when verbose parameter is set."""
    model = BoringModel()
    model.example_input_array = torch.randn(5, 32)
    file_path = os.path.join(tmp_path, "model.onnx")

    with patch("torch.onnx.log", autospec=True) as mocked:
        model.to_onnx(file_path, verbose=True)
    (prefix, _), _ = mocked.call_args
    assert prefix == "Exported graph: "


@RunIf(onnx=True)
def test_error_if_no_input(tmp_path):
    """Test that an error is thrown when there is no input tensor."""
    model = BoringModel()
    model.example_input_array = None
    file_path = os.path.join(tmp_path, "model.onnx")
    with pytest.raises(
        ValueError,
        match=r"Could not export to ONNX since neither `input_sample` nor"
        r" `model.example_input_array` attribute is set.",
    ):
        model.to_onnx(file_path)


@RunIf(onnx=True)
def test_input_check_runs_onnx_checker(tmp_path):
    """`input_check=True` should load the exported model and run `onnx.checker.check_model`."""
    import onnx

    model = BoringModel()
    input_sample = torch.randn((1, 32))

    file_path = os.path.join(tmp_path, "model.onnx")
    model.to_onnx(file_path, input_sample, input_check=True)
    assert os.path.isfile(file_path)
    # Sanity-check: same file should also pass onnx.checker when loaded independently.
    onnx.checker.check_model(onnx.load(file_path))

    # BytesIO path: the cursor position should be unchanged after the check reads the buffer.
    buf = BytesIO()
    model.to_onnx(file_path=buf, input_sample=input_sample, input_check=True)
    end_pos = buf.tell()
    assert end_pos > 4e2
    assert len(buf.getvalue()) == end_pos


@RunIf(onnx=True)
def test_input_check_raises_without_file_path():
    """`input_check=True` needs a path or BytesIO to load the exported model from."""
    model = BoringModel()
    model.example_input_array = torch.randn((1, 32))
    with pytest.raises(ValueError, match=r"`input_check=True` requires `file_path`"):
        model.to_onnx(file_path=None, input_check=True)


@RunIf(onnx=True)
def test_input_check_detects_invalid_model(tmp_path, monkeypatch):
    """If the saved file isn't a valid ONNX model, `input_check=True` should raise."""
    import onnx

    model = BoringModel()
    input_sample = torch.randn((1, 32))
    file_path = os.path.join(tmp_path, "model.onnx")

    def _raise(_):
        raise onnx.checker.ValidationError("forced failure")

    monkeypatch.setattr(onnx.checker, "check_model", _raise)
    with pytest.raises(onnx.checker.ValidationError, match="forced failure"):
        model.to_onnx(file_path, input_sample, input_check=True)


@RunIf(onnx=True, min_torch="2.5.0", dynamo=True, onnxscript=True)
def test_input_check_rejects_dynamo():
    """`input_check=True` is not compatible with the dynamo exporter."""
    model = BoringModel()
    model.example_input_array = torch.randn((1, 32))
    with pytest.raises(ValueError, match=r"`input_check=True` is not supported together with `dynamo=True`"):
        model.to_onnx(input_check=True, dynamo=True)


@pytest.mark.parametrize(
    "dynamo",
    [
        None,
        pytest.param(False, marks=RunIf(min_torch="2.5.0", dynamo=True, onnxscript=True)),
        pytest.param(True, marks=RunIf(min_torch="2.5.0", dynamo=True, onnxscript=True)),
    ],
)
@RunIf(onnx=True)
def test_if_inference_output_is_valid(tmp_path, dynamo):
    """Test that the output inferred from ONNX model is same as from PyTorch."""
    model = BoringModel()
    model.example_input_array = torch.randn(5, 32)

    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)

    model.eval()
    with torch.no_grad():
        torch_out = model(model.example_input_array)

    file_path = os.path.join(tmp_path, "model.onnx")
    kwargs = {
        "export_params": True,
    }
    if dynamo is not None:
        kwargs["dynamo"] = dynamo
    model.to_onnx(file_path, model.example_input_array, **kwargs)

    ort_kwargs = {"providers": "CPUExecutionProvider"} if compare_version("onnxruntime", operator.ge, "1.16.0") else {}
    ort_session = onnxruntime.InferenceSession(file_path, **ort_kwargs)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(model.example_input_array)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    assert np.allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


@RunIf(min_torch="2.5.0", dynamo=True)
@pytest.mark.skipif(_ONNXSCRIPT_AVAILABLE, reason="Run this test only if onnxscript is not available.")
def test_model_onnx_export_missing_onnxscript():
    """Test that an error is raised if onnxscript is not available."""
    model = BoringModel()
    model.example_input_array = torch.randn(5, 32)

    with pytest.raises(
        ModuleNotFoundError,
        match=re.escape(
            f"`{type(model).__name__}.to_onnx(dynamo=True)` requires `onnxscript` and `torch>=2.5.0` to be installed.",
        ),
    ):
        model.to_onnx(dynamo=True)


@RunIf(onnx=True, min_torch="2.5.0", dynamo=True, onnxscript=True)
def test_model_return_type():
    if _TORCH_GREATER_EQUAL_2_6:
        from torch.onnx import ONNXProgram
    else:
        from torch.onnx._internal.exporter import ONNXProgram

    model = BoringModel()
    model.example_input_array = torch.randn((1, 32))
    model.eval()

    onnx_pg = model.to_onnx(dynamo=True)
    assert isinstance(onnx_pg, ONNXProgram)

    model_ret = model(model.example_input_array)
    inf_ret = onnx_pg(model.example_input_array)
    assert torch.allclose(model_ret, inf_ret[0], rtol=1e-03, atol=1e-05)


@RunIf(max_torch="2.5.0")
def test_model_onnx_export_wrong_torch_version():
    """Test that an error is raised if onnxscript is not available."""
    model = BoringModel()
    model.example_input_array = torch.randn(5, 32)

    with pytest.raises(
        ModuleNotFoundError,
        match=re.escape(
            f"`{type(model).__name__}.to_onnx(dynamo=True)` requires `onnxscript` and `torch>=2.5.0` to be installed.",
        ),
    ):
        model.to_onnx(dynamo=True)
