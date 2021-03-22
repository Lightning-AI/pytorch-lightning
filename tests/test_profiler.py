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
import logging
import os
import platform
import time
from distutils.version import LooseVersion
from pathlib import Path

import numpy as np
import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_8
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf

PROFILER_OVERHEAD_MAX_TOLERANCE = 0.0005


def _get_python_cprofile_total_duration(profile):
    return sum([x.inlinetime for x in profile.getstats()])


def _sleep_generator(durations):
    """
    the profile_iterable method needs an iterable in which we can ensure that we're
    properly timing how long it takes to call __next__
    """
    for duration in durations:
        time.sleep(duration)
        yield duration


@pytest.fixture
def simple_profiler():
    return SimpleProfiler()


@pytest.mark.parametrize(["action", "expected"], [
    pytest.param("a", [3, 1]),
    pytest.param("b", [2]),
    pytest.param("c", [1]),
])
def test_simple_profiler_durations(simple_profiler, action: str, expected: list):
    """Ensure the reported durations are reasonably accurate."""

    for duration in expected:
        with simple_profiler.profile(action):
            time.sleep(duration)

    # different environments have different precision when it comes to time.sleep()
    # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/796
    np.testing.assert_allclose(simple_profiler.recorded_durations[action], expected, rtol=0.2)


@pytest.mark.parametrize(["action", "expected"], [
    pytest.param("a", [3, 1]),
    pytest.param("b", [2]),
    pytest.param("c", [1]),
])
def test_simple_profiler_iterable_durations(simple_profiler, action: str, expected: list):
    """Ensure the reported durations are reasonably accurate."""
    iterable = _sleep_generator(expected)

    for _ in simple_profiler.profile_iterable(iterable, action):
        pass

    # we exclude the last item in the recorded durations since that's when StopIteration is raised
    np.testing.assert_allclose(simple_profiler.recorded_durations[action][:-1], expected, rtol=0.2)


def test_simple_profiler_overhead(simple_profiler, n_iter=5):
    """Ensure that the profiler doesn't introduce too much overhead during training."""
    for _ in range(n_iter):
        with simple_profiler.profile("no-op"):
            pass

    durations = np.array(simple_profiler.recorded_durations["no-op"])
    assert all(durations < PROFILER_OVERHEAD_MAX_TOLERANCE)


def test_simple_profiler_describe(caplog, simple_profiler):
    """Ensure the profiler won't fail when reporting the summary."""
    with caplog.at_level(logging.INFO):
        simple_profiler.describe()

    assert "Profiler Report" in caplog.text


def test_simple_profiler_value_errors(simple_profiler):
    """Ensure errors are raised where expected."""

    action = "test"
    with pytest.raises(ValueError):
        simple_profiler.stop(action)

    simple_profiler.start(action)

    with pytest.raises(ValueError):
        simple_profiler.start(action)

    simple_profiler.stop(action)


@pytest.fixture
def advanced_profiler(tmpdir):
    return AdvancedProfiler(output_filename=os.path.join(tmpdir, "profiler.txt"))


@pytest.mark.parametrize(["action", "expected"], [
    pytest.param("a", [3, 1]),
    pytest.param("b", [2]),
    pytest.param("c", [1]),
])
def test_advanced_profiler_durations(advanced_profiler, action: str, expected: list):

    for duration in expected:
        with advanced_profiler.profile(action):
            time.sleep(duration)

    # different environments have different precision when it comes to time.sleep()
    # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/796
    recored_total_duration = _get_python_cprofile_total_duration(advanced_profiler.profiled_actions[action])
    expected_total_duration = np.sum(expected)
    np.testing.assert_allclose(recored_total_duration, expected_total_duration, rtol=0.2)


@pytest.mark.parametrize(["action", "expected"], [
    pytest.param("a", [3, 1]),
    pytest.param("b", [2]),
    pytest.param("c", [1]),
])
def test_advanced_profiler_iterable_durations(advanced_profiler, action: str, expected: list):
    """Ensure the reported durations are reasonably accurate."""
    iterable = _sleep_generator(expected)

    for _ in advanced_profiler.profile_iterable(iterable, action):
        pass

    recored_total_duration = _get_python_cprofile_total_duration(advanced_profiler.profiled_actions[action])
    expected_total_duration = np.sum(expected)
    np.testing.assert_allclose(recored_total_duration, expected_total_duration, rtol=0.2)


def test_advanced_profiler_overhead(advanced_profiler, n_iter=5):
    """
    ensure that the profiler doesn't introduce too much overhead during training
    """
    for _ in range(n_iter):
        with advanced_profiler.profile("no-op"):
            pass

    action_profile = advanced_profiler.profiled_actions["no-op"]
    total_duration = _get_python_cprofile_total_duration(action_profile)
    average_duration = total_duration / n_iter
    assert average_duration < PROFILER_OVERHEAD_MAX_TOLERANCE


def test_advanced_profiler_describe(tmpdir, advanced_profiler):
    """
    ensure the profiler won't fail when reporting the summary
    """
    # record at least one event
    with advanced_profiler.profile("test"):
        pass
    # log to stdout and print to file
    advanced_profiler.describe()
    data = Path(advanced_profiler.output_fname).read_text()
    assert len(data) > 0


def test_advanced_profiler_value_errors(advanced_profiler):
    """Ensure errors are raised where expected."""

    action = "test"
    with pytest.raises(ValueError):
        advanced_profiler.stop(action)

    advanced_profiler.start(action)
    advanced_profiler.stop(action)


@pytest.fixture
def pytorch_profiler(tmpdir):
    return PyTorchProfiler(output_filename=os.path.join(tmpdir, "profiler.txt"), local_rank=0)


@pytest.mark.skipif(_TORCH_GREATER_EQUAL_1_8, reason="This feature isn't support with PyTorch 1.8 profiler")
def test_pytorch_profiler_describe(pytorch_profiler):
    """Ensure the profiler won't fail when reporting the summary."""
    pytorch_profiler.start("on_test_start")
    with pytorch_profiler.profile("test_step"):
        pass

    # log to stdout and print to file
    pytorch_profiler.describe()
    data = Path(pytorch_profiler.output_fname).read_text()
    assert len(data) > 0


def test_pytorch_profiler_value_errors(pytorch_profiler):
    """Ensure errors are raised where expected."""
    action = "test_step"
    pytorch_profiler.start(action)
    pytorch_profiler.stop(action)

    with pytest.raises(MisconfigurationException, match="profiled_functions` and `PyTorchProfiler.record"):
        PyTorchProfiler(profiled_functions=["a"], record_functions=["b"])
    pytorch_profiler.teardown()


@RunIf(min_gpus=2, special=True)
def test_pytorch_profiler_trainer_ddp(tmpdir, pytorch_profiler):
    """Ensure that the profiler can be given to the training and default step are properly recorded. """
    model = BoringModel()
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        limit_train_batches=6,
        limit_val_batches=6,
        profiler=pytorch_profiler,
        accelerator="ddp",
        gpus=2,
        logger=TensorBoardLogger(tmpdir)
    )
    trainer.fit(model)

    if not _TORCH_GREATER_EQUAL_1_8:
        data = Path(pytorch_profiler.output_fname).read_text()
        assert len(data) > 0
    else:
        files = os.listdir(trainer.profiler.path_to_export_trace)
        files = sorted([file for file in files if file.endswith('.json')])
        if os.getenv("LOCAL_RANK", "0") == "0":
            assert 'training_step_and_backward_0' in files[0]
            assert 'validation_step_0' in files[2]
        else:
            assert 'training_step_and_backward_1' in files[1]
            assert 'validation_step_1' in files[3]


def test_pytorch_profiler_trainer_fit(tmpdir, pytorch_profiler):
    """Ensure that the profiler can be given to the trainer and training, validation steps are properly recorded. """
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=5,
        limit_val_batches=5,
        profiler=pytorch_profiler,
    )
    trainer.fit(model)

    if not _TORCH_GREATER_EQUAL_1_8:
        expected = ('validation_step', 'training_step_and_backward', 'training_step', 'backward')
        for name in expected:
            assert len([e for e in pytorch_profiler.function_events if name == e.name]) > 0
        data = Path(pytorch_profiler.output_fname).read_text()
        assert len(data) > 0
    else:
        files = os.listdir(tmpdir if pytorch_profiler == PyTorchProfiler else trainer.profiler.path_to_export_trace)
        files = sorted([file for file in files if file.endswith('.json')])
        assert 'training_step_and_backward_0' in files[0]
        assert 'validation_step_0' in files[1]
        assert len(files) == 2


def test_pytorch_profiler_trainer_test(tmpdir):
    """Ensure that the profiler can be given to the trainer and test step are properly recorded. """
    pytorch_profiler = PyTorchProfiler(
        output_filename=os.path.join(tmpdir, "profiler.txt"), local_rank=0, path_to_export_trace=tmpdir
    )
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_test_batches=10,
        profiler=pytorch_profiler,
    )
    trainer.test(model)

    if not _TORCH_GREATER_EQUAL_1_8:
        assert len([e for e in pytorch_profiler.function_events if 'test_step' == e.name]) > 0
        data = Path(pytorch_profiler.output_fname).read_text()
        assert len(data) > 0
    else:
        files = sorted([file for file in os.listdir(tmpdir) if file.endswith('.json')])
        assert 'test_step_0' in files[0]


def test_pytorch_profiler_trainer_predict(tmpdir):
    """Ensure that the profiler can be given to the trainer and predict function are properly recorded. """
    pytorch_profiler = PyTorchProfiler(
        output_filename=os.path.join(tmpdir, "profiler.txt"), local_rank=0, path_to_export_trace=tmpdir
    )
    model = BoringModel()
    model.predict_dataloader = model.train_dataloader
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_test_batches=2,
        profiler=pytorch_profiler,
    )
    trainer.predict(model)

    if not _TORCH_GREATER_EQUAL_1_8:
        assert len([e for e in pytorch_profiler.function_events if 'predict' == e.name]) > 0
        data = Path(pytorch_profiler.output_fname).read_text()
        assert len(data) > 0
    else:
        files = sorted([file for file in os.listdir(tmpdir) if file.endswith('.json')])
        assert 'predict_0' in files[0]


@RunIf(min_gpus=1, special=True)
@pytest.mark.skipif(_TORCH_GREATER_EQUAL_1_8, reason="This feature isn't support with PyTorch 1.8 profiler")
def test_pytorch_profiler_nested_emit_nvtx(tmpdir):
    """
    This test check emit_nvtx is correctly supported
    """
    pytorch_profiler = PyTorchProfiler(use_cuda=True, emit_nvtx=True)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        profiler=pytorch_profiler,
        gpus=1,
    )
    trainer.fit(model)


def test_pytorch_profiler_nested(tmpdir):
    """Ensure that the profiler handles nested context"""

    pytorch_profiler = PyTorchProfiler(
        export_to_chrome=False,
        record_functions=["a", "b", "c"],
        use_cuda=torch.cuda.is_available(),
        output_filename=os.path.join(tmpdir, "profiler.txt")
    )

    with pytorch_profiler.profile("a"):
        a = torch.ones(42)
        with pytorch_profiler.profile("b"):
            b = torch.zeros(42)
        with pytorch_profiler.profile("c"):
            _ = a + b

    pytorch_profiler.describe()

    events_name = {e.name for e in pytorch_profiler.function_events}

    if platform.system() == "Windows":
        expected = {'a', 'add', 'b', 'c', 'profiler::_record_function_enter', 'profiler::_record_function_exit'}
    else:
        expected = {
            'signed char', 'add', 'profiler::_record_function_exit', 'bool', 'char', 'profiler::_record_function_enter'
        }

    if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
        expected = {'add', 'zeros', 'ones', 'zero_', 'b', 'fill_', 'c', 'a', 'empty'}

    if LooseVersion(torch.__version__) >= LooseVersion("1.7.0"):
        expected = {
            'aten::zeros', 'aten::add', 'aten::zero_', 'c', 'b', 'a', 'aten::fill_', 'aten::empty', 'aten::ones'
        }

    assert events_name == expected, (events_name, torch.__version__, platform.system())


@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_8, reason="Need at least PyTorch 1.8")
@pytest.mark.parametrize('profiler', ('pytorch', PyTorchProfiler))
def test_pytorch_profiler_trainer_new_api(tmpdir, profiler):
    """Ensure that the profiler can be given to the training and default step are properly recorded. """

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=10,
        profiler=profiler if isinstance(profiler, str) else profiler(path_to_export_trace=tmpdir),
    )
    trainer.fit(model)

    files = os.listdir(tmpdir if profiler == PyTorchProfiler else trainer.profiler.path_to_export_trace)
    files = sorted([file for file in files if file.endswith('.json')])
    assert 'training_step_and_backward_0' in files[0]
    assert 'validation_step_0' in files[1]
    assert len(files) == 2
