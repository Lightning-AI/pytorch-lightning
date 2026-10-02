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
import pytest
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.profilers.pytorch import _KINETO_AVAILABLE, ScheduleWrapper

pytestmark = pytest.mark.skipif(not _KINETO_AVAILABLE, reason="Requires PyTorch Profiler Kineto")


@pytest.mark.parametrize("stage", ["training_step", "validation_step", "test_step", "predict_step"])
@pytest.mark.parametrize("repeat", [0, 1, 3])
def test_schedule_wrapper_repeated_cycles(stage, repeat):
    schedule = torch.profiler.schedule(wait=2, warmup=1, active=3, repeat=repeat)
    wrapper = ScheduleWrapper(schedule)
    expected = [schedule(step) for step in range(1, 25)]
    for _ in range(2):
        wrapper.setup("on_validation_start" if stage == "validation_step" else "on_fit_start")
        wrapper.pre_step(stage)
        assert [wrapper(step) for step in range(24)] == expected
        wrapper.reset()


def test_schedule_wrapper_repeated_train_validation_cycles():
    schedule = torch.profiler.schedule(wait=2, warmup=1, active=3, repeat=2)
    wrapper = ScheduleWrapper(schedule)
    wrapper.setup("on_fit_start")
    # Sanity validation must not advance the validation schedule.
    wrapper.pre_step("validation_step")
    assert [wrapper(step) for step in range(2)] == [schedule(0)] * 2
    assert wrapper.num_step == 0

    for cycle in range(3):
        expected = [schedule(step) for step in range(cycle * 6 + 1, (cycle + 1) * 6 + 1)]
        for stage in ("training_step", "validation_step"):
            wrapper.pre_step(stage)
            assert [wrapper(step) for step in range(6)] == expected


@pytest.mark.parametrize("trainer_fn", ["fit", "validate", "test", "predict"])
@pytest.mark.parametrize("use_default_schedule", [False, True])
def test_pytorch_profiler_schedule_trace_cycles(tmp_path, trainer_fn, use_default_schedule):
    traces = []
    schedule_kwargs = (
        {} if use_default_schedule else {"schedule": torch.profiler.schedule(wait=2, warmup=1, active=3, repeat=5)}
    )
    profiler = PyTorchProfiler(
        dirpath=tmp_path,
        filename="repeat",
        activities=[torch.profiler.ProfilerActivity.CPU],
        on_trace_ready=lambda p: traces.append(p.step_num),
        export_to_chrome=False,
        record_module_names=False,
        **schedule_kwargs,
    )
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=32,
        limit_val_batches=0 if trainer_fn == "fit" else 32,
        limit_test_batches=32,
        limit_predict_batches=32,
        num_sanity_val_steps=0,
        profiler=profiler,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    getattr(trainer, trainer_fn)(BoringModel())
    assert traces == ([5] if use_default_schedule else [6, 12, 18, 24, 30])
