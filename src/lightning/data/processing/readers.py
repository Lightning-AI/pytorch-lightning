import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

from lightning_utilities.core.imports import RequirementCache

from lightning.data.utilities.env import _DistributedEnv
from lightning.data.utilities.shuffle import _associate_chunks_and_internals_to_ranks

_POLARS_AVAILABLE = RequirementCache("polars")
_PYARROW_AVAILABLE = RequirementCache("pyarrow")


class BaseReader(ABC):

    def get_num_nodes(self) -> int:
        return int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 1))

    def get_node_rank(self) -> int:
        return int(os.getenv("DATA_OPTIMIZER_NODE_RANK", 0))

    @abstractmethod
    def items_to_workers(self, items: List[Any], num_workers: int) -> List[List[Any]]:
        """This method is meant to convert the items provided by the users into items to be processed by the
        workers."""
        pass

    @abstractmethod
    def read(self, item: Any) -> Any:
        """Read the data associated to an item."""
        pass


@dataclass
class ParquetSlice:
    """Keep track of a parquet file slice with its filepath, start and end."""
    filepath: str
    start: int
    end: int


class ParquetReader(BaseReader):

    def __init__(self, num_rows: Optional[int] = 2048, to_pandas: bool = True) -> None:
        self.num_rows = num_rows
        self.to_pandas = to_pandas

        if not _PYARROW_AVAILABLE or not _POLARS_AVAILABLE:
            raise ModuleNotFoundError("Please, run: `pip install pyarrow polars`")

    def _get_num_rows(self, path: str) -> int:
        if _PYARROW_AVAILABLE:
            import pyarrow.dataset as ds
            df = ds.dataset(path).scanner()
            return df.count_rows()

        # FIXED: There is a bug in polars. This leads to read_parquet to hang.
        if _POLARS_AVAILABLE:
            import polars as pol
            df = pol.scan_parquet(path)
            num_rows = df.select(pol.len()).collect().item()
            return num_rows

        raise RuntimeError("Please, install either pyarrow or polars.")

    def read(self, item: ParquetSlice) -> Any:
        if _POLARS_AVAILABLE:
            import polars as pol
            df = pol.scan_parquet(item.filepath).slice(item.start, item.end).collect()

            if self.to_pandas:
                df = df.to_pandas()

            return df

        if _PYARROW_AVAILABLE:
            import pyarrow.dataset as ds

            df = ds.dataset(item.filepath).scanner()

            df = df.take([item.start, item.end])

            if self.to_pandas:
                df.to_pandas()

            return df

        raise RuntimeError("Please, install either pyarrow or polars.")


    def items_to_workers(self, items: Any, num_workers: int) -> List[List[ParquetSlice]]:
        intervals = [(0, self._get_num_rows(item)) for item in items]

        world_size = self.get_num_nodes() * num_workers
        node_rank = self.get_node_rank()

        fake_distributed_env = _DistributedEnv(world_size, 0, self.get_num_nodes())
        parquet_indexes_per_worker, p_slices_per_worker = _associate_chunks_and_internals_to_ranks(
            fake_distributed_env, list(range(len(items))), intervals, False)

        workers_user_items: List[List[ParquetSlice]] = [[] for _ in range(num_workers)]

        iterator = enumerate(zip(parquet_indexes_per_worker, p_slices_per_worker))

        node_start = node_rank * num_workers
        node_end = (node_rank + 1) * num_workers

        for worker_idx, (parquet_indexes, p_slices) in iterator:
            if node_start <= worker_idx < node_end:
                if self.num_rows:
                    workers_user_items[worker_idx % num_workers].extend([
                        ParquetSlice(
                            items[parquet_index], p_slice_start, p_slice_start + self.num_rows
                            if p_slice[1] > (p_slice_start + self.num_rows) else
                            p_slice[1]
                        )
                        for parquet_index, p_slice in zip(parquet_indexes, p_slices)
                        for p_slice_start in range(p_slice[0], p_slice[1] + self.num_rows, self.num_rows)
                        if p_slice_start < p_slice[1]
                    ])
                else:
                    workers_user_items[worker_idx % num_workers].extend([
                        ParquetSlice(items[parquet_index], *p_slice)
                        for parquet_index, p_slice in zip(parquet_indexes, p_slices)
                    ])

        assert len(workers_user_items) == num_workers
        assert all(len(w) for w in workers_user_items)

        return workers_user_items
