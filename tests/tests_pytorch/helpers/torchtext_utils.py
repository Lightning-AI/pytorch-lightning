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
import random
import string

from tests_pytorch.helpers.imports import Dataset, Example, Field, Iterator


def _generate_random_string(length: int = 10):
    return "".join(random.choices(string.ascii_letters, k=length))


def get_dummy_torchtext_data_iterator(num_samples: int, batch_size: int, include_lengths: bool = False):
    text_field = Field(
        sequential=True,
        pad_first=False,  # nosec
        init_token="<s>",
        eos_token="</s>",  # nosec
        include_lengths=include_lengths,
    )  # nosec

    dataset = Dataset(
        [
            Example.fromdict({"text": _generate_random_string()}, {"text": ("text", text_field)})
            for _ in range(num_samples)
        ],
        {"text": text_field},
    )
    text_field.build_vocab(dataset)

    iterator = Iterator(
        dataset,
        batch_size=batch_size,
        sort_key=None,
        device=None,
        batch_size_fn=None,
        train=True,
        repeat=False,
        shuffle=None,
        sort=None,
        sort_within_batch=None,
    )
    return iterator, text_field
