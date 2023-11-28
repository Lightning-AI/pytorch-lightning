# Copyright The Lightning AI team.
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

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from lightning.data.constants import _TORCH_GREATER_EQUAL_2_1_0
from lightning.data.utilities.env import _get_node_rank, _get_num_nodes

if _TORCH_GREATER_EQUAL_2_1_0:
    from torch.utils._pytree import tree_flatten


def _pack_greedily(items: List[Any], weights: List[int], num_bins: int) -> Tuple[Dict[int, List[Any]], Dict[int, int]]:
    """Greedily pack items with given weights into bins such that the total weight of each bin is roughly equally
    distributed among all bins."""

    if len(items) != len(weights):
        raise ValueError(f"Items and weights must have the same length, got {len(items)} and {len(weights)}.")
    if any(w <= 0 for w in weights):
        raise ValueError("All weights must be positive.")

    sorted_items_and_weights = sorted(zip(items, weights), key=lambda x: x[1], reverse=True)
    bin_contents = defaultdict(list)
    bin_weights = {i: 0 for i in range(num_bins)}

    for item, weight in sorted_items_and_weights:
        min_bin_id = min(bin_weights, key=(lambda x: bin_weights[x]), default=0)
        bin_contents[min_bin_id].append(item)
        bin_weights[min_bin_id] += weight

    return bin_contents, bin_weights


def _get_item_filesizes(items: List[Any], base_path: str = "") -> List[int]:
    """Computes the total size in bytes of all file paths for every datastructure in the given list."""
    item_sizes = []
    for item in items:
        flattened_item, _ = tree_flatten(item)

        num_bytes = 0
        for element in flattened_item:
            if isinstance(element, str) and element.startswith(base_path) and os.path.exists(element):
                file_bytes = os.path.getsize(element)
                if file_bytes == 0:
                    raise RuntimeError(f"The file {element} has 0 bytes!")
                num_bytes += file_bytes
        item_sizes.append(num_bytes)
    return item_sizes


def _sort_greedily(items: List[Any], weights: List[int]) -> List[Any]:
    """Greedily sort items based on their weights."""

    if len(items) != len(weights):
        raise ValueError(f"Items and weights must have the same length, got {len(items)} and {len(weights)}.")
    if any(w <= 0 for w in weights):
        raise ValueError("All weights must be positive.")

    return [item for (item, _) in sorted(zip(items, weights), key=lambda x: x[1], reverse=True)]


def _chunk_list(seq: List[Any], size: int) -> List[List[Any]]:
    """Split a list of items into sub list of equal sizes."""
    return [seq[pos : pos + size] for pos in range(0, len(seq), size)]


def _map_items_to_workers_sequentially(num_workers: int, user_items: List[Any]) -> List[List[Any]]:
    num_nodes = _get_num_nodes()
    current_node_rank = _get_node_rank()
    node_size = len(user_items) // num_nodes
    workers_user_items = []
    for node_rank in range(num_nodes):
        if node_rank != current_node_rank:
            continue
        is_last_node = node_rank == num_nodes - 1
        start_node = node_rank * node_size
        end_node = len(user_items) if is_last_node else (node_rank + 1) * node_size
        node_user_items = user_items[start_node:end_node]
        worker_size = len(node_user_items) // num_workers
        for worker_idx in range(num_workers):
            is_last = worker_idx == num_workers - 1
            begin = worker_idx * worker_size
            end = len(node_user_items) if is_last else (worker_idx + 1) * worker_size
            workers_user_items.append(node_user_items[begin:end])
    return workers_user_items


def _map_items_to_workers_weighted(
    num_workers: int, user_items: List[Any], weights: Optional[List[int]] = None
) -> List[List[Any]]:
    # Associate the items to the workers based on number of nodes and node rank.
    weights = [1] * len(user_items) if weights is None else weights
    num_nodes = _get_num_nodes()
    node_rank = _get_node_rank()
    world_size = num_nodes * num_workers

    worker_items, worker_weights = _pack_greedily(items=user_items, weights=weights, num_bins=world_size)
    worker_ids_this_node = range(node_rank * num_workers, (node_rank + 1) * num_workers)

    for worker_id, size in worker_weights.items():
        if worker_id not in worker_ids_this_node:
            continue
        print(f"Worker {worker_id} gets {size / 1e6:.1f} MB ({len(worker_items[worker_id])} files)")

    return [worker_items[worker_id] for worker_id in worker_ids_this_node]
