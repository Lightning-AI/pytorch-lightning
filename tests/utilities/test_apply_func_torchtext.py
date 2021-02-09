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
import pytest
import torch
import torchtext
from torchtext.data.example import Example

from pytorch_lightning.utilities.apply_func import move_data_to_device


def _get_torchtext_data_iterator(include_lengths=False):
    text_field = torchtext.data.Field(
        sequential=True,
        pad_first=False,  # nosec
        init_token="<s>",
        eos_token="</s>",  # nosec
        include_lengths=include_lengths
    )  # nosec

    example1 = Example.fromdict({"text": "a b c a c"}, {"text": ("text", text_field)})
    example2 = Example.fromdict({"text": "b c a a"}, {"text": ("text", text_field)})
    example3 = Example.fromdict({"text": "c b a"}, {"text": ("text", text_field)})

    dataset = torchtext.data.Dataset(
        [example1, example2, example3],
        {"text": text_field},
    )
    text_field.build_vocab(dataset)

    iterator = torchtext.data.Iterator(
        dataset,
        batch_size=3,
        sort_key=None,
        device=None,
        batch_size_fn=None,
        train=True,
        repeat=False,
        shuffle=None,
        sort=None,
        sort_within_batch=None
    )
    return iterator, text_field


@pytest.mark.parametrize('include_lengths', [False, True])
@pytest.mark.parametrize(['device'], [pytest.param(torch.device('cuda', 0))])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test assumes GPU machine")
def test_batch_move_data_to_device_torchtext_include_lengths(include_lengths, device):
    data_iterator, _ = _get_torchtext_data_iterator(include_lengths=include_lengths)
    data_iter = iter(data_iterator)
    batch = next(data_iter)
    batch_on_device = move_data_to_device(batch, device)

    if include_lengths:
        # tensor with data
        assert (batch_on_device.text[0].device == device)
        # tensor with length of data
        assert (batch_on_device.text[1].device == device)
    else:
        assert (batch_on_device.text.device == device)


@pytest.mark.parametrize('include_lengths', [False, True])
def test_batch_move_data_to_device_torchtext_include_lengths_cpu(include_lengths):
    test_batch_move_data_to_device_torchtext_include_lengths(include_lengths, torch.device('cpu'))
