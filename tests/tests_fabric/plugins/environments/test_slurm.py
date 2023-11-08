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
import logging
import os
import shutil
import sys
from unittest import mock

import pytest
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning_utilities.test.warning import no_warning_call

from tests_fabric.helpers.runif import RunIf


@mock.patch.dict(os.environ, {}, clear=True)
def test_default_attributes():
    """Test the default attributes when no environment variables are set."""
    env = SLURMEnvironment()
    assert env.creates_processes_externally
    assert env.main_address == "127.0.0.1"
    assert env.main_port == 12910
    assert env.job_name() is None
    assert env.job_id() is None

    with pytest.raises(KeyError):
        # world size is required to be passed as env variable
        env.world_size()
    with pytest.raises(KeyError):
        # local rank is required to be passed as env variable
        env.local_rank()
    with pytest.raises(KeyError):
        # node_rank is required to be passed as env variable
        env.node_rank()


@mock.patch.dict(
    os.environ,
    {
        "SLURM_NODELIST": "1.1.1.1, 1.1.1.2",
        "SLURM_JOB_ID": "0001234",
        "SLURM_NTASKS": "20",
        "SLURM_NTASKS_PER_NODE": "10",
        "SLURM_LOCALID": "2",
        "SLURM_PROCID": "1",
        "SLURM_NODEID": "3",
        "SLURM_JOB_NAME": "JOB",
    },
)
def test_attributes_from_environment_variables(caplog):
    """Test that the SLURM cluster environment takes the attributes from the environment variables."""
    env = SLURMEnvironment()
    assert env.auto_requeue is True
    assert env.main_address == "1.1.1.1"
    assert env.main_port == 15000 + 1234
    assert env.job_id() == int("0001234")
    assert env.world_size() == 20
    assert env.global_rank() == 1
    assert env.local_rank() == 2
    assert env.node_rank() == 3
    assert env.job_name() == "JOB"
    # setter should be no-op
    with caplog.at_level(logging.DEBUG, logger="lightning.fabric.plugins.environments"):
        env.set_global_rank(100)
    assert env.global_rank() == 1
    assert "setting global rank is not allowed" in caplog.text

    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger="lightning.fabric.plugins.environments"):
        env.set_world_size(100)
    assert env.world_size() == 20
    assert "setting world size is not allowed" in caplog.text


@pytest.mark.parametrize(
    ("slurm_node_list", "expected"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("alpha", "alpha"),
        ("alpha,beta,gamma", "alpha"),
        ("alpha beta gamma", "alpha"),
        ("1.2.3.[100-110]", "1.2.3.100"),
        ("1.2.3.[089, 100-110]", "1.2.3.089"),
        ("host[22]", "host22"),
        ("host[1,5-9]", "host1"),
        ("host[5-9,1]", "host5"),
        ("alpha, host[5-9], gamma", "alpha"),
        ("alpha[3,1], beta", "alpha3"),
    ],
)
def test_main_address_from_slurm_node_list(slurm_node_list, expected):
    """Test extracting the main node from different formats for the SLURM_NODELIST."""
    with mock.patch.dict(os.environ, {"SLURM_NODELIST": slurm_node_list}):
        env = SLURMEnvironment()
        assert env.main_address == expected


def test_main_address_and_port_from_env_variable():
    env = SLURMEnvironment()
    with mock.patch.dict(os.environ, {"MASTER_ADDR": "1.2.3.4", "MASTER_PORT": "1234"}):
        assert env.main_address == "1.2.3.4"
        assert env.main_port == 1234


def test_detect():
    """Test the detection of a SLURM environment configuration."""
    with mock.patch.dict(os.environ, {}, clear=True):
        assert not SLURMEnvironment.detect()

    with mock.patch.dict(os.environ, {"SLURM_NTASKS": "2"}):
        assert SLURMEnvironment.detect()

    with mock.patch.dict(os.environ, {"SLURM_JOB_NAME": "bash"}):
        assert not SLURMEnvironment.detect()

    with mock.patch.dict(os.environ, {"SLURM_JOB_NAME": "interactive"}):
        assert not SLURMEnvironment.detect()


@RunIf(skip_windows=True)
@pytest.mark.skipif(shutil.which("srun") is not None, reason="must run on a machine where srun is not available")
def test_srun_available_and_not_used(monkeypatch):
    """Test that a warning is emitted if Lightning suspects the user forgot to run their script with `srun`."""
    monkeypatch.setattr(sys, "argv", ["train.py", "--lr", "0.01"])
    expected = "`srun` .* available .* but is not used. HINT: .* srun python train.py --lr 0.01"

    # pretend `srun` is available
    with mock.patch("lightning.fabric.plugins.environments.slurm.shutil.which", return_value="/usr/bin/srun"):
        with pytest.warns(PossibleUserWarning, match=expected):
            SLURMEnvironment()

        with pytest.warns(PossibleUserWarning, match=expected):
            SLURMEnvironment.detect()

    # no warning if `srun` is unavailable
    with no_warning_call(PossibleUserWarning, match=expected):
        SLURMEnvironment()
        assert not SLURMEnvironment.detect()


def test_srun_variable_validation():
    """Test that we raise useful errors when `srun` variables are misconfigured."""
    with mock.patch.dict(os.environ, {"SLURM_NTASKS": "1"}):
        SLURMEnvironment()
    with mock.patch.dict(os.environ, {"SLURM_NTASKS": "2"}), pytest.raises(
        RuntimeError, match="You set `--ntasks=2` in your SLURM"
    ):
        SLURMEnvironment()


@mock.patch.dict(os.environ, {"SLURM_NTASKS_PER_NODE": "4", "SLURM_NNODES": "2"})
def test_validate_user_settings():
    """Test that the environment can validate the number of devices and nodes set in Fabric/Trainer."""
    env = SLURMEnvironment()
    env.validate_settings(num_devices=4, num_nodes=2)

    with pytest.raises(ValueError, match="the number of tasks per node configured .* does not match"):
        env.validate_settings(num_devices=2, num_nodes=2)

    with pytest.raises(ValueError, match="the number of nodes configured in SLURM .* does not match"):
        env.validate_settings(num_devices=4, num_nodes=1)

    # in interactive mode, validation is skipped becauses processes get launched by Fabric/Trainer, not SLURM
    with mock.patch(
        "lightning.fabric.plugins.environments.slurm.SLURMEnvironment.job_name", return_value="interactive"
    ):
        env = SLURMEnvironment()
        env.validate_settings(num_devices=4, num_nodes=1)  # no error
