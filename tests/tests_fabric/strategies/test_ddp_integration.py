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
from copy import deepcopy

import pytest
import torch
from lightning.fabric import Fabric

from tests_fabric.helpers.runif import RunIf
from tests_fabric.strategies.test_single_device import _run_test_clip_gradients


@pytest.mark.parametrize(
    "accelerator",
    [
        "cpu",
        pytest.param("cuda", marks=RunIf(min_cuda_gpus=2)),
    ],
)
def test_ddp_save_load(accelerator, tmp_path):
    """Test that DDP model checkpoints can be saved and loaded successfully."""
    fabric = Fabric(devices=2, accelerator=accelerator, strategy="ddp_spawn")
    fabric.launch(_run_ddp_save_load, tmp_path)


def _run_ddp_save_load(fabric, tmp_path):
    fabric.seed_everything(0)

    tmp_path = fabric.broadcast(tmp_path)

    model = torch.nn.Linear(2, 2)
    params_before = deepcopy(list(model.parameters()))

    # Save
    fabric.save(tmp_path / "saved_before_setup.ckpt", {"model": model})
    wrapped_model = fabric.setup(model)
    fabric.save(tmp_path / "saved_after_setup.ckpt", {"model": wrapped_model})

    def assert_params_equal(params0, params1):
        assert all(torch.equal(p0, p1.to(p0.device)) for p0, p1 in zip(params0, params1))

    # Load
    model = torch.nn.Linear(2, 2)
    fabric.load(tmp_path / "saved_before_setup.ckpt", {"model": model})
    assert_params_equal(params_before, model.parameters())
    fabric.load(tmp_path / "saved_after_setup.ckpt", {"model": model})
    assert_params_equal(params_before, model.parameters())

    wrapped_model = fabric.setup(model)
    fabric.load(tmp_path / "saved_before_setup.ckpt", {"model": wrapped_model})
    assert_params_equal(params_before, wrapped_model.parameters())
    fabric.load(tmp_path / "saved_after_setup.ckpt", {"model": wrapped_model})
    assert_params_equal(params_before, wrapped_model.parameters())


@pytest.mark.parametrize(
    ("clip_type", "accelerator", "precision"),
    [
        ("norm", "cpu", "32-true"),
        ("val", "cpu", "32-true"),
        ("norm", "cpu", "bf16-mixed"),
        ("val", "cpu", "bf16-mixed"),
        pytest.param("norm", "cuda", "32-true", marks=RunIf(min_cuda_gpus=2)),
        pytest.param("val", "cuda", "32-true", marks=RunIf(min_cuda_gpus=2)),
        pytest.param("norm", "cuda", "16-mixed", marks=RunIf(min_cuda_gpus=2)),
        pytest.param("val", "cuda", "16-mixed", marks=RunIf(min_cuda_gpus=2)),
        pytest.param("norm", "cuda", "bf16-mixed", marks=RunIf(min_cuda_gpus=2, bf16_cuda=True)),
        pytest.param("val", "cuda", "bf16-mixed", marks=RunIf(min_cuda_gpus=2, bf16_cuda=True)),
    ],
)
@RunIf(standalone=True)
def test_clip_gradients(clip_type, accelerator, precision):
    if clip_type == "norm" and precision == "16-mixed":
        pytest.skip(reason="Clipping by norm with 16-mixed is numerically unstable.")

    fabric = Fabric(accelerator=accelerator, devices=2, precision=precision, strategy="ddp")
    fabric.launch()
    _run_test_clip_gradients(fabric=fabric, clip_type=clip_type)
