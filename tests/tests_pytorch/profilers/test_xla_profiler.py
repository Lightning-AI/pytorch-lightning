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
import os
from multiprocessing import Event, Process
from unittest import mock

import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.profilers import XLAProfiler

from tests_pytorch.helpers.runif import RunIf


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_xla_profiler_instance(tmp_path):
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True, profiler="xla", accelerator="tpu", devices="auto")

    assert isinstance(trainer.profiler, XLAProfiler)
    trainer.fit(model)


@pytest.mark.xfail(strict=False, reason="XLA Profiler doesn't support Prog. capture yet")
def test_xla_profiler_prog_capture(tmp_path):
    import torch_xla.debug.profiler as xp
    import torch_xla.utils.utils as xu

    port = xu.get_free_tcp_ports()[0]
    training_started = Event()

    def train_worker():
        model = BoringModel()
        trainer = Trainer(default_root_dir=tmp_path, max_epochs=4, profiler="xla", accelerator="tpu", devices=8)

        trainer.fit(model)

    p = Process(target=train_worker, daemon=True)
    p.start()
    training_started.wait(120)

    logdir = str(tmp_path)
    xp.trace(f"localhost:{port}", logdir, duration_ms=2000, num_tracing_attempts=5, delay_ms=1000)

    p.terminate()

    assert os.isfile(os.path.join(logdir, "plugins", "profile", "*", "*.xplane.pb"))
