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
"""Test deprecated functionality which will be removed in v1.6.0"""
import pytest

from pytorch_lightning.plugins.training_type import DDPPlugin, DDPSpawnPlugin


def test_v1_6_0_ddp_num_nodes():
    with pytest.deprecated_call(match="Argument `num_nodes` in `DDPPlugin` is deprecated in v1.4"):
        DDPPlugin(num_nodes=1)


def test_v1_6_0_ddp_sync_batchnorm():
    with pytest.deprecated_call(match="Argument `sync_batchnorm` in `DDPPlugin` is deprecated in v1.4"):
        DDPPlugin(sync_batchnorm=False)


def test_v1_6_0_ddp_spawn_num_nodes():
    with pytest.deprecated_call(match="Argument `num_nodes` in `DDPPlugin` is deprecated in v1.4"):
        DDPSpawnPlugin(num_nodes=1)


def test_v1_6_0_ddp_spawn_sync_batchnorm():
    with pytest.deprecated_call(match="Argument `sync_batchnorm` in `DDPPlugin` is deprecated in v1.4"):
        DDPSpawnPlugin(sync_batchnorm=False)
