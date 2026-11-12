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

import itertools
import math
import pickle

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.combined_loader import CombinedLoader


def _reference(mode, rows):
    if mode == "sequential":
        return [
            (value, index, loader_index)
            for loader_index, values in enumerate(rows)
            for index, value in enumerate(values)
        ]
    if mode == "min_size":
        pairs = zip(*rows)
    elif mode == "max_size":
        pairs = itertools.zip_longest(*rows)
    elif any(not values for values in rows):
        # Preserve max_size_cycle's existing behavior for an empty iterable.
        pairs = []
    else:
        pairs = itertools.islice(zip(*(itertools.cycle(values) for values in rows)), max(map(len, rows)))
    return [({"a": left, "b": right}, index, 0) for index, (left, right) in enumerate(pairs)]


def _collect(loader):
    results = []
    # Bound collection so the original broken implementation is also safe to run
    # against infinite iterable datasets.
    for batch, index, loader_index in itertools.islice(loader, 10):
        if isinstance(batch, dict):
            batch = {key: value.item() if value is not None else None for key, value in batch.items()}
        else:
            batch = batch.item()
        results.append((batch, index, loader_index))
    return results


@pytest.mark.parametrize("mode", ["min_size", "max_size", "max_size_cycle", "sequential"])
@pytest.mark.parametrize("limits", [[2, 1], [math.inf, 2], [10, 10], None])
def test_combined_loader_limits_match_truncated_input_sequences(mode, limits):
    rows = [list(range(5)), list(range(10, 14))]
    loader = CombinedLoader({"a": DataLoader(rows[0]), "b": DataLoader(rows[1])}, mode)
    loader.limits = limits
    limited = (
        [values[: int(limit)] if math.isfinite(limit) else values for values, limit in zip(rows, limits)]
        if limits
        else rows
    )
    expected = _reference(mode, limited)
    for _ in range(2):
        iter(loader)
        assert len(loader) == len(expected)
        assert _collect(loader) == expected
        loader.reset()


class _InfiniteNumbers(IterableDataset):
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        yield from itertools.count(self.start)


@pytest.mark.parametrize("mode", ["min_size", "max_size", "max_size_cycle"])
@pytest.mark.parametrize("limits", [[3, 2], [1, 4]])
def test_limits_bound_infinite_iterable_dataloaders(mode, limits):
    loader = CombinedLoader({"a": DataLoader(_InfiniteNumbers(0)), "b": DataLoader(_InfiniteNumbers(10))}, mode)
    loader.limits = limits
    expected = _reference(mode, [list(range(limits[0])), list(range(10, 10 + limits[1]))])
    iter(loader)
    assert len(loader) == len(expected)
    assert _collect(loader) == expected


@pytest.mark.parametrize("mode", ["min_size", "max_size", "max_size_cycle", "sequential"])
def test_zero_limit_behaves_like_an_empty_iterator(mode):
    loader = CombinedLoader({"a": DataLoader(range(5)), "b": DataLoader(range(10, 14))}, mode)
    loader.limits = [3, 0]
    assert _collect(loader) == _reference(mode, [list(range(3)), []])


class _LimitedTrainingModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.seen = []

    def training_step(self, batch, batch_idx):
        self.seen.append((batch["a"].item(), batch["b"].item()))
        return (self.weight * (batch["a"] + batch["b"]).float()).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0001)

    def train_dataloader(self):
        return {"a": DataLoader(range(8)), "b": DataLoader(range(10, 14))}


@pytest.mark.parametrize("fraction", [0.5, 0.75])
@pytest.mark.parametrize("epochs", [1, 2])
def test_trainer_fractional_limits_cycle_only_allowed_batches(tmp_path, fraction, epochs):
    model = _LimitedTrainingModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        max_epochs=epochs,
        limit_train_batches=fraction,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    left_limit, right_limit = int(8 * fraction), int(4 * fraction)
    expected_pairs = [(index, 10 + index % right_limit) for index in range(left_limit)] * epochs
    assert model.seen == expected_pairs
    reference = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([reference], lr=0.0001)
    for left, right in expected_pairs:
        optimizer.zero_grad()
        (reference * (left + right)).square().backward()
        optimizer.step()
    torch.testing.assert_close(model.weight, reference, rtol=0, atol=0)


def test_limited_loader_remains_pickleable_after_starting():
    loader = CombinedLoader({"a": DataLoader(range(5)), "b": DataLoader(range(10, 14))}, "max_size_cycle")
    loader.limits = [2, 1]
    next(iter(loader))
    restored = pickle.loads(pickle.dumps(loader))
    assert _collect(restored) == _reference("max_size_cycle", [[0, 1], [10]])
