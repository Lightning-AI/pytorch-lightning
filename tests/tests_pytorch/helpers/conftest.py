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
import urllib.request

import pytest
import torch

from tests_pytorch.helpers.datasets import MNIST


@pytest.fixture
def mock_mnist_download(monkeypatch):
    """Use generated MNIST-shaped data while exercising the dataset download and cache paths."""
    generator = torch.Generator().manual_seed(0)
    splits = {}
    for filename, num_samples in ((MNIST.TRAIN_FILE_NAME, 1200), (MNIST.TEST_FILE_NAME, 200)):
        images = torch.randint(256, (num_samples, 28, 28), dtype=torch.uint8, generator=generator)
        targets = torch.arange(10).repeat(num_samples // 10)
        splits[filename] = (images, targets)

    def download(url, filename):
        assert url in MNIST.RESOURCES
        torch.save(splits[url.rsplit("/", 1)[-1]], filename)
        return filename, None

    monkeypatch.setattr(urllib.request, "urlretrieve", download)
    return splits
