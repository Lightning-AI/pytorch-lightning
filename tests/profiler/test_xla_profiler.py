# Copyright The PyTorch Lightning team.
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

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.profiler import XLAProfiler
from pytorch_lightning.utilities import _TPU_AVAILABLE
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf

if _TPU_AVAILABLE:
    import torch_xla.debug.profiler as xp
    import torch_xla.utils.utils as xu


@RunIf(tpu=True)
def test_xla_profiler_instance(tmpdir):

    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, profiler="xla", tpu_cores=8)

    assert isinstance(trainer.profiler, XLAProfiler)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@pytest.mark.skipif(True, reason="XLA Profiler doesn't support Prog. capture yet")
def test_xla_profiler_prog_capture(tmpdir):

    port = xu.get_free_tcp_ports()[0]
    training_started = Event()

    def train_worker():
        model = BoringModel()
        trainer = Trainer(default_root_dir=tmpdir, max_epochs=4, profiler="xla", tpu_cores=8)

        trainer.fit(model)

    p = Process(target=train_worker, daemon=True)
    p.start()
    training_started.wait(120)

    logdir = str(tmpdir)
    xp.trace(f"localhost:{port}", logdir, duration_ms=2000, num_tracing_attempts=5, delay_ms=1000)

    p.terminate()

    assert os.isfile(os.path.join(logdir, "plugins", "profile", "*", "*.xplane.pb"))
