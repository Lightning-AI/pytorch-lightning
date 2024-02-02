import os
from abc import ABC, abstractmethod
from lightning_utilities.core.imports import RequirementCache
from typing import Any, List, Optional
from dataclasses import dataclass
from lightning.data.streaming.shuffle import _associate_chunks_and_internals_to_ranks
from lightning.data.utilities.env import _DistributedEnv


_POLARS_AVAILABLE = RequirementCache("polars")
_PYARROW_AVAILABLE = RequirementCache("pyarrow")


class BaseReader(ABC):

    def get_num_nodes(self) -> int:
        return int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 1))

    @abstractmethod
    def to_workers_user_items(self, items: List[Any], num_workers: int) -> List[List[Any]]:
        pass

    @abstractmethod
    def read(self, item: Any) -> Any:
        pass


@dataclass
class ParquetSlice:
    filepath: str
    start: int
    end: int


class ParquetReader(BaseReader):

    def __init__(self, num_rows: Optional[int] = 2048, to_pandas: bool = True) -> None:
        self.num_rows = num_rows
        self.to_pandas = to_pandas

    def _get_num_rows(self, path: str) -> int:
        # TODO: There is a bug in polars. This leads to read_parquet to hang.
        # if _POLARS_AVAILABLE:
        #     import polars as pol
        #     df = pol.scan_parquet(path)
        #     num_rows = df.select(pol.len()).collect().item()
        #     return num_rows

        if _PYARROW_AVAILABLE:
            import pyarrow.dataset as ds
            df = ds.dataset(path).scanner()
            return df.count_rows()

        raise RuntimeError("Please, install either pyarrow or polars.")

    def read(self, item: ParquetSlice) -> Any:
        if _POLARS_AVAILABLE:
            import polars as pol
            df = pol.read_parquet(item.filepath, row_index_offset=item.start, n_rows=item.end - item.start, parallel="row_groups")

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
        

    def to_workers_user_items(self, items: Any, num_workers: int) -> List[List[ParquetSlice]]:
        intervals = [(0, self._get_num_rows(item)) for item in items]
        
        world_size = self.get_num_nodes() * num_workers

        fake_distributed_env = _DistributedEnv(world_size, 0, self.get_num_nodes())
        parquet_indexes_per_worker, parquet_slices_per_worker = _associate_chunks_and_internals_to_ranks(fake_distributed_env, list(range(len(items))), intervals, False)

        workers_user_items = [[] for _ in range(world_size)]

        for worker_idx, (parquet_indexes, parquet_slices) in enumerate(zip(parquet_indexes_per_worker, parquet_slices_per_worker)):
            if self.num_rows:
                workers_user_items[worker_idx].extend([
                    ParquetSlice(
                        items[parquet_index], parquet_slice_start, parquet_slice_start + self.num_rows 
                        if parquet_slice[1] > (parquet_slice_start + self.num_rows) else  
                        parquet_slice[1]
                    )
                    for parquet_index, parquet_slice in zip(parquet_indexes, parquet_slices)
                    for parquet_slice_start in range(parquet_slice[0], parquet_slice[1] + self.num_rows, self.num_rows)
                    if parquet_slice_start < parquet_slice[1]
                ])
            else:
                workers_user_items[worker_idx].extend([
                    ParquetSlice(items[parquet_index], *parquet_slice)
                    for parquet_index, parquet_slice in zip(parquet_indexes, parquet_slices)
                ])

        return workers_user_items
