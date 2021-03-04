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
from distutils.version import LooseVersion
from pathlib import Path

import numpy as np
import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
from pytorch_lightning.utilities.exceptions import MisconfigurationException
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
    profiler = SimpleProfiler()
    return profiler


@pytest.mark.parametrize(["action", "expected"], [
    pytest.param("a", [3, 1]),
    pytest.param("b", [2]),
    pytest.param("c", [1]),
])
def test_simple_profiler_durations(simple_profiler, action, expected):
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
def test_simple_profiler_iterable_durations(simple_profiler, action, expected):
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
    profiler = AdvancedProfiler(output_filename=os.path.join(tmpdir, "profiler.txt"))
    return profiler


@pytest.mark.parametrize(["action", "expected"], [
    pytest.param("a", [3, 1]),
    pytest.param("b", [2]),
    pytest.param("c", [1]),
])
def test_advanced_profiler_durations(advanced_profiler, action, expected):

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
def test_advanced_profiler_iterable_durations(advanced_profiler, action, expected):
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
    profiler = PyTorchProfiler(output_filename=os.path.join(tmpdir, "profiler.txt"), local_rank=0)
    return profiler


def test_pytorch_profiler_describe(pytorch_profiler):
    """Ensure the profiler won't fail when reporting the summary."""
    with pytorch_profiler.profile("test_step"):
        pass

    # log to stdout and print to file
    pytorch_profiler.describe()
    data = Path(pytorch_profiler.output_fname).read_text()
    assert len(data) > 0


def test_pytorch_profiler_value_errors(pytorch_profiler):
    """Ensure errors are raised where expected."""

    action = "test_step"
    with pytest.raises(ValueError):
        pytorch_profiler.stop(action)

    pytorch_profiler.start(action)
    pytorch_profiler.stop(action)

    with pytest.raises(MisconfigurationException, match="profiled_functions` and `PyTorchProfiler.record"):
        PyTorchProfiler(profiled_functions=[], record_functions=[])


@RunIf(min_gpus=2, special=True)
def test_pytorch_profiler_trainer_ddp(tmpdir):
    """Ensure that the profiler can be given to the training and default step are properly recorded. """

    output_filename = os.path.join(tmpdir, "profiler.txt")

    profiler = PyTorchProfiler(output_filename=output_filename)

    model = BoringModel()
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        limit_train_batches=6,
        limit_val_batches=6,
        profiler=profiler,
        accelerator="ddp",
        gpus=2,
        logger=TensorBoardLogger(tmpdir)
    )
    trainer.fit(model)

    assert len(profiler.summary()) > 0
    profiler.describe()
    data = Path(profiler.output_fname).read_text()
    assert len(data) > 0


@pytest.mark.parametrize("use_output_filename", [True])
def test_pytorch_profiler_trainer(tmpdir, use_output_filename):
    """Ensure that the profiler can be given to the training and default step are properly recorded. """

    if use_output_filename:
        output_filename = os.path.join(tmpdir, "profiler.txt")
    else:
        output_filename = None

    profiler = PyTorchProfiler(output_filename=output_filename)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        profiler=profiler,
    )
    trainer.fit(model)

    enabled = use_output_filename or not use_output_filename and profiler.local_rank == 0

    if enabled:
        assert len(profiler.summary()) > 0
        expected = {'validation_step', 'training_step_and_backward', 'training_step', 'backward'}
        assert set(profiler.profiled_actions.keys()) == expected
    else:
        assert profiler.summary() is None
        assert set(profiler.profiled_actions.keys()) == set()

    if use_output_filename:
        profiler.describe()
        data = Path(profiler.output_fname).read_text()
        assert len(data) > 0


def test_pytorch_profiler_nested(tmpdir):
    """Ensure that the profiler handles nested context"""

    pytorch_profiler = PyTorchProfiler(
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
