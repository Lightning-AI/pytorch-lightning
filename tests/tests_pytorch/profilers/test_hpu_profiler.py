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
from pathlib import Path

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.profilers import AdvancedProfiler, HPUProfiler, SimpleProfiler
from pytorch_lightning.utilities import _HPU_AVAILABLE
from tests_pytorch.helpers.runif import RunIf

if _HPU_AVAILABLE:
    import habana_frameworks
else:
    raise RuntimeError("Expected device type HPU for running HPU Profiling ...")


class TestHPUProfiler:
    def setup_method(self):
        try:
            shutil.rmtree("profiler_logs")
        except:
            pass

    def teardown_method(self):
        try:
            shutil.rmtree("profiler_logs")
        except:
            pass

    @pytest.fixture
    def get_device_count(self, pytestconfig):
        hpus = int(pytestconfig.getoption("hpus"))
        if not hpus:
            assert habana_frameworks.torch.hpu.device_count() >= 1
            return 1
        assert (hpus <= habana_frameworks.torch.hpu.device_count(), "More hpu devices asked than present")
        assert hpus == 1 or hpus % 8 == 0
        return hpus

    @RunIf(hpu=True)
    def test_hpu_simple_profiler_instances(self, tmpdir, get_device_count):
        model = BoringModel()
        trainer = Trainer(
            profiler="simple", accelerator="hpu", devices=get_device_count, max_epochs=1, default_root_dir=tmpdir
        )
        assert isinstance(trainer.profiler, SimpleProfiler)
        trainer.fit(model)
        assert trainer.state.finished, f"Training failed with {trainer.state}"

    @RunIf(hpu=True)
    def test_hpu_simple_profiler_trainer_stages(self, tmpdir, get_device_count):
        model = BoringModel()
        profiler = SimpleProfiler(dirpath=os.path.join(tmpdir, "profiler_logs"), filename="profiler")
        trainer = Trainer(
            profiler=profiler, accelerator="hpu", devices=get_device_count, max_epochs=1, default_root_dir=tmpdir
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
            profiler="advanced", accelerator="hpu", devices=get_device_count, max_epochs=1, default_root_dir=tmpdir
        )
        assert isinstance(trainer.profiler, AdvancedProfiler)
        trainer.fit(model)
        assert trainer.state.finished, f"Training failed with {trainer.state}"

    @RunIf(hpu=True)
    def test_hpu_advanced_profiler_trainer_stages(self, tmpdir, get_device_count):
        model = BoringModel()
        profiler = AdvancedProfiler(dirpath=os.path.join(tmpdir, "profiler_logs"), filename="profiler")
        trainer = Trainer(
            profiler=profiler, accelerator="hpu", devices=get_device_count, max_epochs=1, default_root_dir=tmpdir
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
    def test_simple_profiler_distributed_files(self, tmpdir, get_device_count):
        """Ensure the proper files are saved in distributed."""
        profiler = SimpleProfiler(dirpath="profiler_logs", filename="profiler")
        model = BoringModel()
        trainer = Trainer(
            default_root_dir=tmpdir,
            fast_dev_run=2,
            strategy="hpu_parallel",
            accelerator="hpu",
            devices=get_device_count,
            profiler=profiler,
            logger=False,
        )
        trainer.fit(model)
        trainer.validate(model)
        trainer.test(model)

        expected = {
            f"{stage}-profiler-{rank}.txt"
            for stage in ("fit", "validate", "test")
            for rank in range(0, trainer.num_devices)
        }
        actual = set(os.listdir(profiler.dirpath))
        print(f"dirpath: {profiler.dirpath}; actual: {actual}; expected: {expected}")
        assert actual == expected

        for f in os.listdir(profiler.dirpath):
            assert Path(os.path.join(os.getcwd(), profiler.dirpath, f)).read_text("utf-8")

    @RunIf(hpu=True)
    def test_advanced_profiler_distributed_files(self, tmpdir, get_device_count):
        """Ensure the proper files are saved in distributed."""
        profiler = AdvancedProfiler(dirpath="profiler_logs", filename="profiler")
        model = BoringModel()
        trainer = Trainer(
            default_root_dir=tmpdir,
            fast_dev_run=2,
            strategy="hpu_parallel",
            accelerator="hpu",
            devices=get_device_count,
            profiler=profiler,
            logger=False,
        )
        trainer.fit(model)
        trainer.validate(model)
        trainer.test(model)

        expected = {
            f"{stage}-profiler-{rank}.txt"
            for stage in ("fit", "validate", "test")
            for rank in range(0, trainer.num_devices)
        }
        actual = set(os.listdir(profiler.dirpath))
        print(f"dirpath: {profiler.dirpath}; actual: {actual}; expected: {expected}")
        assert actual == expected

        for f in os.listdir(profiler.dirpath):
            assert Path(os.path.join(os.getcwd(), profiler.dirpath, f)).read_text("utf-8")

    def test_hpu_pytorch_profiler_instances(tmpdir):
        devices = habana_frameworks.torch.hpu.device_count()
        model = BoringModel()

        trainer = Trainer(profiler="hpu", accelerator="hpu", devices=devices, max_epochs=1)
        assert isinstance(trainer.profiler, HPUProfiler)
        trainer.fit(model)
        assert trainer.state.finished, f"Training failed with {trainer.state}"

    def test_hpu_trace_event_cpu_instant_event(tmpdir):
        # Run model and prep json
        devices = habana_frameworks.torch.hpu.device_count()
        model = BoringModel()
        trainer = Trainer(accelerator="hpu", devices=devices, max_epochs=1, profiler=HPUProfiler(profile_memory=True))
        trainer.fit(model)
        assert trainer.state.finished, f"Training failed with {trainer.state}"

        # get trace path
        TRACE_PATH = glob.glob(os.path.join("lightning_logs", "version_0", "fit*training_step*.json"))[0]

        # Check json dumped
        assert os.path.isfile(TRACE_PATH)
        with open(TRACE_PATH) as file:
            data = json.load(file)
            assert "traceEvents" in data
            event_time_arr = []
            for event in data["traceEvents"]:
                try:
                    if event["cat"] == "cpu_instant_event":
                        event_time_arr.append(event["ts"])
                except KeyError:
                    pass
            if len(event_time_arr) == 0:
                raise Exception("Could not find event cpu_instant_event in trace")
            for event_time in event_time_arr:
                assert event_time >= 0

    def test_hpu_trace_event_python_function(tmpdir):
        # Run model and prep json
        devices = habana_frameworks.torch.hpu.device_count()
        model = BoringModel()

        trainer = Trainer(accelerator="hpu", devices=devices, max_epochs=1, profiler=HPUProfiler(with_stack=True))
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
                    if event["cat"] == "python_function":
                        event_duration_arr.append(event["dur"])
                except KeyError:
                    pass
            if len(event_duration_arr) == 0:
                raise Exception("Could not find event python_function in trace")
            for event_duration in event_duration_arr:
                assert event_duration >= 0

    def test_hpu_trace_event_cpu_op(tmpdir):
        # Run model and prep json
        devices = habana_frameworks.torch.hpu.device_count()
        model = BoringModel()

        trainer = Trainer(accelerator="hpu", devices=devices, max_epochs=1, profiler=HPUProfiler())
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

    def test_hpu_trace_event_hpu_op(tmpdir):
        # Run model and prep json
        devices = habana_frameworks.torch.hpu.device_count()
        model = BoringModel()

        trainer = Trainer(accelerator="hpu", devices=devices, max_epochs=1, profiler=HPUProfiler())
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

    def test_hpu_trace_event_hpu_meta_op(tmpdir):
        # Run model and prep json
        devices = habana_frameworks.torch.hpu.device_count()
        model = BoringModel()

        trainer = Trainer(accelerator="hpu", devices=devices, max_epochs=1, profiler=HPUProfiler())
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

    def test_hpu_trace_event_kernel(tmpdir):
        # Run model and prep json
        devices = habana_frameworks.torch.hpu.device_count()
        model = BoringModel()
        trainer = Trainer(accelerator="hpu", devices=devices, max_epochs=1, profiler=HPUProfiler())
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

    def test_hpu_trace_event_cpu_op_input_dim(tmpdir):
        # Run model and prep json
        devices = habana_frameworks.torch.hpu.device_count()
        model = BoringModel()

        trainer = Trainer(accelerator="hpu", devices=devices, max_epochs=1, profiler=HPUProfiler(record_shapes=True))
        trainer.fit(model)
        assert trainer.state.finished, f"Training failed with {trainer.state}"

        # get trace path
        TRACE_PATH = glob.glob(os.path.join("lightning_logs", "version_0", "fit*training_step*.json"))[0]

        # Check json dumped
        assert os.path.isfile(TRACE_PATH)
        with open(TRACE_PATH) as file:
            data = json.load(file)
            assert "traceEvents" in data
            input_dim_arr = []
            for event in data["traceEvents"]:
                try:
                    if event["cat"] == "cpu_op":
                        input_dim_arr.append(event["args"]["Input Dims"])
                except KeyError:
                    pass
            if len(input_dim_arr) == 0:
                raise Exception("Could not find event Input Dims in trace")
            for input_dim in input_dim_arr:
                assert input_dim is not None

    def test_hpu_trace_event_call_stack(tmpdir):
        # Run model and prep json
        devices = habana_frameworks.torch.hpu.device_count()
        model = BoringModel()

        trainer = Trainer(accelerator="hpu", devices=devices, max_epochs=1, profiler=HPUProfiler(with_stack=True))
        trainer.fit(model)
        assert trainer.state.finished, f"Training failed with {trainer.state}"

        # get trace path
        TRACE_PATH = glob.glob(os.path.join("lightning_logs", "version_0", "fit*training_step*.json"))[0]

        # Check json dumped
        assert os.path.isfile(TRACE_PATH)
        with open(TRACE_PATH) as file:
            data = json.load(file)
            assert "traceEvents" in data
            call_stack_arr = []
            for event in data["traceEvents"]:
                try:
                    if event["cat"] == "cpu_op":
                        call_stack_arr.append(event["args"]["Call stack"])
                except KeyError:
                    pass
            if len(call_stack_arr) == 0:
                raise Exception("Could not find event cpu_op in trace")
            for call_stack in call_stack_arr:
                assert call_stack is not None
