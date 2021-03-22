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
import time
from copy import deepcopy
from distutils.version import LooseVersion

import numpy as np
import pytest
import torch

from pytorch_lightning import Trainer, Callback
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
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


def test_simple_profiler_value_errors(simple_profiler):
    """Ensure errors are raised where expected."""

    action = "test"
    with pytest.raises(ValueError):
        simple_profiler.stop(action)

    simple_profiler.start(action)

    with pytest.raises(ValueError):
        simple_profiler.start(action)

    simple_profiler.stop(action)


def test_simple_profiler_deepcopy(tmpdir):
    simple_profiler = SimpleProfiler(dirpath=tmpdir, filename="test")
    simple_profiler.describe()
    assert deepcopy(simple_profiler)


def test_simple_profiler_log_dir(tmpdir):
    """Ensure the profiler dirpath defaults to `trainer.log_dir` when not present"""
    profiler = SimpleProfiler(filename="profiler")
    assert profiler._log_dir is None

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        profiler=profiler,
    )
    trainer.fit(model)

    expected = tmpdir / "lightning_logs" / "version_0"
    assert trainer.log_dir == expected
    assert profiler._log_dir == trainer.log_dir
    assert expected.join("fit-profiler.txt").exists()


@RunIf(skip_windows=True)
def test_simple_profiler_distributed_files(tmpdir):
    """Ensure the proper files are saved in distributed"""
    profiler = SimpleProfiler(dirpath=tmpdir, filename='profiler')
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=2,
        accelerator="ddp_cpu",
        num_processes=2,
        profiler=profiler,
        logger=False,
    )
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)

    actual = set(os.listdir(profiler.dirpath))
    expected = {f"{stage}-profiler-{rank}.txt" for stage in ("fit", "validate", "test") for rank in (0, 1)}
    assert actual == expected

    for f in profiler.dirpath.listdir():
        assert f.read_text('utf-8')


def test_simple_profiler_logs(tmpdir, caplog, simple_profiler):
    """Ensure that the number of printed logs is correct"""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=2,
        profiler=simple_profiler,
        logger=False,
    )
    with caplog.at_level(logging.INFO, logger="pytorch_lightning.profiler.profilers"):
        trainer.fit(model)
        trainer.test(model)

    assert caplog.text.count("Profiler Report") == 2


@pytest.fixture
def advanced_profiler(tmpdir):
    return AdvancedProfiler(dirpath=tmpdir, filename="profiler")


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
    path = advanced_profiler.dirpath / f"{advanced_profiler.filename}.txt"
    data = path.read_text("utf-8")
    assert len(data) > 0


def test_advanced_profiler_value_errors(advanced_profiler):
    """Ensure errors are raised where expected."""

    action = "test"
    with pytest.raises(ValueError):
        advanced_profiler.stop(action)

    advanced_profiler.start(action)
    advanced_profiler.stop(action)


def test_advanced_profiler_deepcopy(advanced_profiler):
    advanced_profiler.describe()
    assert deepcopy(advanced_profiler)


@pytest.fixture
def pytorch_profiler(tmpdir):
    return PyTorchProfiler(dirpath=tmpdir, filename="profiler")


def test_pytorch_profiler_describe(pytorch_profiler):
    """Ensure the profiler won't fail when reporting the summary."""
    with pytorch_profiler.profile("test_step"):
        pass

    # log to stdout and print to file
    pytorch_profiler.describe()
    path = pytorch_profiler.dirpath / f"{pytorch_profiler.filename}.txt"
    data = path.read_text("utf-8")
    assert len(data) > 0


def test_pytorch_profiler_value_errors(pytorch_profiler):
    """Ensure errors are raised where expected."""

    action = "test_step"
    with pytest.raises(ValueError):
        pytorch_profiler.stop(action)

    pytorch_profiler.start(action)
    pytorch_profiler.stop(action)


@RunIf(min_gpus=2, special=True)
def test_pytorch_profiler_trainer_ddp(tmpdir, pytorch_profiler):
    """Ensure that the profiler can be given to the training and default step are properly recorded. """
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        profiler=pytorch_profiler,
        accelerator="ddp",
        gpus=2,
    )
    trainer.fit(model)

    assert len(pytorch_profiler.summary()) > 0
    assert set(pytorch_profiler.profiled_actions) == {'training_step_and_backward', 'validation_step'}

    actual = set(os.listdir(pytorch_profiler.dirpath))
    expected = {f"fit-profiler-{rank}.txt" for rank in (0, 1)}
    assert actual == expected

    for f in pytorch_profiler.dirpath.listdir():
        assert f.read_text('utf-8')


def test_pytorch_profiler_nested(tmpdir):
    """Ensure that the profiler handles nested context"""

    pytorch_profiler = PyTorchProfiler(
        profiled_functions=["a", "b", "c"], use_cuda=False, dirpath=tmpdir, filename="profiler"
    )

    with pytorch_profiler.profile("a"):
        a = torch.ones(42)
        with pytorch_profiler.profile("b"):
            b = torch.zeros(42)
        with pytorch_profiler.profile("c"):
            _ = a + b

    pa = pytorch_profiler.profiled_actions

    # From PyTorch 1.8.0, less operation are being traced.
    if LooseVersion(torch.__version__) >= LooseVersion("1.8.0"):
        expected_ = {
            'a': ['ones', 'empty', 'fill_', 'zeros', 'empty', 'zero_', 'add'],
            'b': ['zeros', 'empty', 'zero_'],
            'c': ['add'],
        }
    # From PyTorch 1.6.0, more operation are being traced.
    elif LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
        expected_ = {
            'a': ['ones', 'empty', 'fill_', 'zeros', 'empty', 'zero_', 'fill_', 'add', 'empty'],
            'b': ['zeros', 'empty', 'zero_', 'fill_'],
            'c': ['add', 'empty'],
        }
    else:
        expected_ = {
            'a': ['add'],
            'b': [],
            'c': ['add'],
        }

    for n in ('a', 'b', 'c'):
        pa[n] = [e.name for e in pa[n]]
        if LooseVersion(torch.__version__) >= LooseVersion("1.7.1"):
            pa[n] = [e.replace("aten::", "") for e in pa[n]]
        assert pa[n] == expected_[n]


@RunIf(min_gpus=1, special=True)
def test_pytorch_profiler_nested_emit_nvtx(tmpdir):
    """
    This test check emit_nvtx is correctly supported
    """
    profiler = PyTorchProfiler(use_cuda=True, emit_nvtx=True)

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        profiler=profiler,
        gpus=1,
    )
    trainer.fit(model)


@pytest.mark.parametrize("cls", (SimpleProfiler, AdvancedProfiler, PyTorchProfiler))
def test_profiler_teardown(tmpdir, cls):
    """
    This test checks if profiler teardown method is called when trainer is exiting.
    """
    class TestCallback(Callback):
        def on_fit_end(self, trainer, *args, **kwargs) -> None:
            # describe sets it to None
            assert trainer.profiler._output_file is None

    profiler = cls(dirpath=tmpdir, filename="profiler")
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, profiler=profiler, callbacks=[TestCallback()])
    trainer.fit(model)

    assert profiler._output_file is None


def test_pytorch_profiler_deepcopy(pytorch_profiler):
    pytorch_profiler.describe()
    assert deepcopy(pytorch_profiler)
