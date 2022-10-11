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

import glob
import json
import os
import shutil

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import HPUAccelerator
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.profilers import AdvancedProfiler, HPUProfiler, SimpleProfiler
from tests_pytorch.helpers.runif import RunIf


class TestHPUProfiler:
    def setup_method(self):
        shutil.rmtree("profiler_logs", ignore_errors=True)

    def teardown_method(self):
        shutil.rmtree("profiler_logs", ignore_errors=True)

    @pytest.fixture
    def get_device_count(self, pytestconfig):
        hpus = int(pytestconfig.getoption("hpus"))
        if not hpus:
            assert HPUAccelerator.auto_device_count() >= 1
            return 1
        assert hpus <= HPUAccelerator.auto_device_count(), "More hpu devices asked than present"
        assert hpus == 1 or hpus % 8 == 0
        return hpus

    @RunIf(hpu=True)
    def test_hpu_simple_profiler_instances(self, tmpdir, get_device_count):
        model = BoringModel()
        trainer = Trainer(
            profiler="simple",
            accelerator="hpu",
            devices=get_device_count,
            max_epochs=1,
            default_root_dir=tmpdir,
            fast_dev_run=True,
        )
        assert isinstance(trainer.profiler, SimpleProfiler)

    @RunIf(hpu=True)
    def test_hpu_simple_profiler_trainer_stages(self, tmpdir, get_device_count):
        model = BoringModel()
        profiler = SimpleProfiler(dirpath=os.path.join(tmpdir, "profiler_logs"), filename="profiler")
        trainer = Trainer(
            profiler=profiler,
            accelerator="hpu",
            devices=get_device_count,
            max_epochs=1,
            default_root_dir=tmpdir,
            fast_dev_run=True,
        )

        trainer.fit(model)
        trainer.validate(model)
        trainer.test(model)
        trainer.predict(model)

        actual = set(os.listdir(profiler.dirpath))
        expected = {f"{stage}-profiler.txt" for stage in ("fit", "validate", "test", "predict")}
        assert actual == expected
        for file in list(os.listdir(profiler.dirpath)):
            assert os.path.getsize(os.path.join(profiler.dirpath, file)) > 0

    @RunIf(hpu=True)
    def test_hpu_advanced_profiler_instances(self, tmpdir, get_device_count):
        model = BoringModel()
        trainer = Trainer(
            profiler="advanced",
            accelerator="hpu",
            devices=get_device_count,
            max_epochs=1,
            default_root_dir=tmpdir,
            fast_dev_run=True,
        )
        assert isinstance(trainer.profiler, AdvancedProfiler)

    @RunIf(hpu=True)
    def test_hpu_advanced_profiler_trainer_stages(self, tmpdir, get_device_count):
        model = BoringModel()
        profiler = AdvancedProfiler(dirpath=os.path.join(tmpdir, "profiler_logs"), filename="profiler")
        trainer = Trainer(
            profiler=profiler,
            accelerator="hpu",
            devices=get_device_count,
            max_epochs=1,
            default_root_dir=tmpdir,
            fast_dev_run=True,
        )

        trainer.fit(model)
        trainer.validate(model)
        trainer.test(model)
        trainer.predict(model)

        actual = set(os.listdir(profiler.dirpath))
        expected = {f"{stage}-profiler.txt" for stage in ("fit", "validate", "test", "predict")}
        assert actual == expected
        for file in list(os.listdir(profiler.dirpath)):
            assert os.path.getsize(os.path.join(profiler.dirpath, file)) > 0

    @RunIf(hpu=True)
    def test_hpu_pytorch_profiler_instances(tmpdir):
        model = BoringModel()

        trainer = Trainer(profiler="hpu", accelerator="hpu", devices=1, max_epochs=1, fast_dev_run=True)
        assert isinstance(trainer.profiler, HPUProfiler)

    @RunIf(hpu=True)
    def test_hpu_trace_event_cpu_op(tmpdir):
        # Run model and prep json
        model = BoringModel()

        trainer = Trainer(accelerator="hpu", devices=1, max_epochs=1, profiler=HPUProfiler())
        trainer.fit(model)
        assert trainer.state.finished, f"Training failed with {trainer.state}"

        # get trace path
        TRACE_PATH = glob.glob(os.path.join("lightning_logs", "version_0", "fit*training_step*.json"))[0]

        # Check json dumped
        assert os.path.isfile(TRACE_PATH)
        with open(TRACE_PATH) as file:
            data = json.load(file)
            assert "traceEvents" in data
            event_duration_arr = []
            for event in data["traceEvents"]:
                try:
                    if event["cat"] == "cpu_op":
                        event_duration_arr.append(event["dur"])
                except KeyError:
                    pass
            if len(event_duration_arr) == 0:
                raise Exception("Could not find event cpu_op in trace")
            for event_duration in event_duration_arr:
                assert event_duration >= 0

    @RunIf(hpu=True)
    def test_hpu_trace_event_hpu_op(tmpdir):
        # Run model and prep json
        model = BoringModel()

        trainer = Trainer(accelerator="hpu", devices=1, max_epochs=1, profiler=HPUProfiler())
        trainer.fit(model)
        assert trainer.state.finished, f"Training failed with {trainer.state}"

        # get trace path
        TRACE_PATH = glob.glob(os.path.join("lightning_logs", "version_0", "fit*training_step*.json"))[0]
        # Check json dumped
        assert os.path.isfile(TRACE_PATH)
        with open(TRACE_PATH) as file:
            data = json.load(file)
            assert "traceEvents" in data
            event_duration_arr = []
            for event in data["traceEvents"]:
                try:
                    if event["cat"] == "hpu_op":
                        event_duration_arr.append(event["dur"])
                except KeyError:
                    pass
            if len(event_duration_arr) == 0:
                raise Exception("Could not find event hpu_op in trace")
            for event_duration in event_duration_arr:
                assert event_duration >= 0

    @RunIf(hpu=True)
    def test_hpu_trace_event_hpu_meta_op(tmpdir):
        # Run model and prep json
        model = BoringModel()

        trainer = Trainer(accelerator="hpu", devices=1, max_epochs=1, profiler=HPUProfiler())
        trainer.fit(model)
        assert trainer.state.finished, f"Training failed with {trainer.state}"

        # get trace path
        TRACE_PATH = glob.glob(os.path.join("lightning_logs", "version_0", "fit*training_step*.json"))[0]

        # Check json dumped
        assert os.path.isfile(TRACE_PATH)
        with open(TRACE_PATH) as file:
            data = json.load(file)
            assert "traceEvents" in data
            event_duration_arr = []
            for event in data["traceEvents"]:
                try:
                    if event["cat"] == "hpu_meta_op":
                        event_duration_arr.append(event["dur"])
                except KeyError:
                    pass
            if len(event_duration_arr) == 0:
                raise Exception("Could not find event hpu_meta_op in trace")
            for event_duration in event_duration_arr:
                assert event_duration >= 0

    @RunIf(hpu=True)
    def test_hpu_trace_event_kernel(tmpdir):
        # Run model and prep json
        model = BoringModel()
        trainer = Trainer(accelerator="hpu", devices=1, max_epochs=1, profiler=HPUProfiler())
        trainer.fit(model)
        assert trainer.state.finished, f"Training failed with {trainer.state}"
        # get trace path
        TRACE_PATH = glob.glob(os.path.join("lightning_logs", "version_0", "fit*training_step*.json"))[0]

        # Check json dumped
        assert os.path.isfile(TRACE_PATH)
        with open(TRACE_PATH) as file:
            data = json.load(file)
            assert "traceEvents" in data
            event_duration_arr = []
            for event in data["traceEvents"]:
                try:
                    if event["cat"] == "kernel":
                        event_duration_arr.append(event["dur"])
                except KeyError:
                    pass
            if len(event_duration_arr) == 0:
                raise Exception("Could not find event kernel in trace")
            for event_duration in event_duration_arr:
                assert event_duration >= 0
