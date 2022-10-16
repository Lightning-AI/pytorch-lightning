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
import json
import os

import pytest
from tests_lite.helpers.runif import RunIf

from lightning_lite.accelerators import CPUAccelerator
from lightning_lite.strategies import DeepSpeedStrategy


@pytest.fixture
def deepspeed_config():
    return {
        "optimizer": {"type": "SGD", "params": {"lr": 3e-5}},
        "scheduler": {
            "type": "WarmupLR",
            "params": {"last_batch_iteration": -1, "warmup_min_lr": 0, "warmup_max_lr": 3e-5, "warmup_num_steps": 100},
        },
    }


@pytest.fixture
def deepspeed_zero_config(deepspeed_config):
    return {**deepspeed_config, "zero_allow_untested_optimizer": True, "zero_optimization": {"stage": 2}}


@RunIf(deepspeed=True)
def test_deepspeed_only_compatible_with_cuda():
    """Test that the DeepSpeed strategy raises an exception if an invalid accelerator is used."""
    strategy = DeepSpeedStrategy(accelerator=CPUAccelerator())
    with pytest.raises(RuntimeError, match="The DeepSpeed strategy is only supported on CUDA GPUs"):
        strategy.setup_environment()


@RunIf(deepspeed=True)
def test_deepspeed_with_invalid_config_path():
    """Test to ensure if we pass an invalid config path we throw an exception."""

    with pytest.raises(
        FileNotFoundError, match="You passed in a path to a DeepSpeed config but the path does not exist"
    ):
        DeepSpeedStrategy(config="invalid_path.json")


@RunIf(deepspeed=True)
def test_deepspeed_with_env_path(tmpdir, monkeypatch, deepspeed_config):
    """Test to ensure if we pass an env variable, we load the config from the path."""
    config_path = os.path.join(tmpdir, "temp.json")
    with open(config_path, "w") as f:
        f.write(json.dumps(deepspeed_config))
    monkeypatch.setenv("PL_DEEPSPEED_CONFIG_PATH", config_path)
    strategy = DeepSpeedStrategy()
    assert strategy.config == deepspeed_config


@RunIf(deepspeed=True)
def test_deepspeed_defaults():
    """Ensure that defaults are correctly set as a config for DeepSpeed if no arguments are passed."""
    strategy = DeepSpeedStrategy()
    assert strategy.config is not None
    assert isinstance(strategy.config["zero_optimization"], dict)


@RunIf(deepspeed=True)
def test_deepspeed_custom_activation_checkpointing_params(tmpdir):
    """Ensure if we modify the activation checkpointing parameters, the deepspeed config contains these changes."""
    ds = DeepSpeedStrategy(
        partition_activations=True,
        cpu_checkpointing=True,
        contiguous_memory_optimization=True,
        synchronize_checkpoint_boundary=True,
    )
    checkpoint_config = ds.config["activation_checkpointing"]
    assert checkpoint_config["partition_activations"]
    assert checkpoint_config["cpu_checkpointing"]
    assert checkpoint_config["contiguous_memory_optimization"]
    assert checkpoint_config["synchronize_checkpoint_boundary"]


@RunIf(deepspeed=True)
def test_deepspeed_config_zero_offload(deepspeed_zero_config):
    """Test the various ways optimizer-offloading can be configured."""

    # default config
    strategy = DeepSpeedStrategy(config=deepspeed_zero_config)
    assert "offload_optimizer" not in strategy.config["zero_optimization"]

    # default config
    strategy = DeepSpeedStrategy()
    assert "offload_optimizer" not in strategy.config["zero_optimization"]

    # default config with `offload_optimizer` argument override
    strategy = DeepSpeedStrategy(offload_optimizer=True)
    assert strategy.config["zero_optimization"]["offload_optimizer"] == {
        "buffer_count": 4,
        "device": "cpu",
        "nvme_path": "/local_nvme",
        "pin_memory": False,
    }

    # externally configured through config
    deepspeed_zero_config["zero_optimization"]["offload_optimizer"] = False
    strategy = DeepSpeedStrategy(config=deepspeed_zero_config)
    assert strategy.config["zero_optimization"]["offload_optimizer"] is False
