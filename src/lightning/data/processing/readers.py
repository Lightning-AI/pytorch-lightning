import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

from lightning_utilities.core.imports import RequirementCache
from tqdm import tqdm

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

    def __init__(self, cache_folder: str, num_rows: Optional[int] = 65536, to_pandas: bool = True) -> None:
        super().__init__()
        self.cache_folder = cache_folder
        self.limit_num_rows = num_rows
        self.to_pandas = to_pandas

        if not _PYARROW_AVAILABLE or not _POLARS_AVAILABLE:
            raise ModuleNotFoundError("Please, run: `pip install pyarrow polars`")

        self.parquet_file = None

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

    def read(self, filepath: str) -> Any:
        import pyarrow as pa
        pa.jemalloc_set_decay_ms(0)

        import pyarrow.parquet as pq

        # close the previous parquet file to release the memory
        if self.parquet_file is not None:
            self.parquet_file.close()
            self.parquet_file = None

        self.parquet_file = pq.ParquetFile(filepath, memory_map=True)
        return self.parquet_file

        if _POLARS_AVAILABLE:
            import polars as pol
            t0 = time()
            df = pol.read_parquet(item.filepath)

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


    def items_to_workers(self, filepaths: Any, num_workers: int) -> List[List[ParquetSlice]]:
        import pyarrow.parquet as pq

        print("Starting resharding the parquet files for optimized processing.")

        new_items = []

        cache_folder = os.path.join(self.cache_folder, f"{self.limit_num_rows}")
        os.makedirs(cache_folder, exist_ok=True)

        for filepath in filepaths:
            num_rows = self._get_num_rows(filepath)

            if num_rows < (self.limit_num_rows * 8):
                new_items.append(filepath)
                continue

            table = None
            parquet_filename = os.path.basename(filepath)

            for start in tqdm(range(0, num_rows, self.limit_num_rows)):
                end = min(start + self.limit_num_rows, num_rows)
                chunk_filepath = os.path.join(cache_folder, f"{start}_{end}_{parquet_filename}")
                new_items.append(chunk_filepath)

                if os.path.exists(chunk_filepath):
                    continue

                if table is None:
                    table = pq.read_table(filepath, memory_map=True)

                pq.write_table(table[start: end], chunk_filepath)

        print("Finished resharding the parquet files for optimized processing.")

        return new_items
