import contextlib
import os
from abc import ABC, abstractmethod
from typing import Any, List

from lightning_utilities.core.imports import RequirementCache
from tqdm import tqdm

_PYARROW_AVAILABLE = RequirementCache("pyarrow")

class BaseReader(ABC):

    def get_num_nodes(self) -> int:
        return int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 1))

    def get_node_rank(self) -> int:
        return int(os.getenv("DATA_OPTIMIZER_NODE_RANK", 0))

    @abstractmethod
    def remap_items(self, items: List[Any], num_workers: int) -> List[Any]:
        """This method is meant to remap the items provided by the users into items more adapted to be distributed."""
        pass

    @abstractmethod
    def read(self, item: Any) -> Any:
        """Read the data associated to an item."""
        pass


class ParquetReader(BaseReader):

    def __init__(self, cache_folder: str, num_rows: int = 65536, to_pandas: bool = True) -> None:
        super().__init__()
        self.cache_folder = cache_folder
        self.num_rows = num_rows
        self.to_pandas = to_pandas



        if not _PYARROW_AVAILABLE:
            raise ModuleNotFoundError("Please, run: `pip install pyarrow`")


        self.parquet_file = None

    def _get_num_rows(self, path: str) -> int:
        import pyarrow.dataset as ds

        df = ds.dataset(path).scanner()
        return df.count_rows()

    def read(self, filepath: str) -> Any:
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Try to force dellocation to avoid memory leak
        with contextlib.suppress(Exception):
            pa.jemalloc_set_decay_ms(0)

        # close the previous parquet file to release the memory
        if self.parquet_file is not None:
            self.parquet_file.close()
            self.parquet_file = None

        self.parquet_file = pq.ParquetFile(filepath, memory_map=True)
        return self.parquet_file

    def remap_items(self, filepaths: List[str], _: int) -> List[str]:
        import pyarrow.parquet as pq

        print("Starting resharding the parquet files for optimized processing.")

        new_items = []

        cache_folder = os.path.join(self.cache_folder, f"{self.num_rows}")
        os.makedirs(cache_folder, exist_ok=True)

        for filepath in filepaths:
            num_rows = self._get_num_rows(filepath)

            table = None
            parquet_filename = os.path.basename(filepath)

            for start in tqdm(range(0, num_rows, self.num_rows)):
                end = min(start + self.num_rows, num_rows)
                chunk_filepath = os.path.join(cache_folder, f"{start}_{end}_{parquet_filename}")
                new_items.append(chunk_filepath)

                if os.path.exists(chunk_filepath):
                    continue

                if table is None:
                    table = pq.read_table(filepath, memory_map=True)

                pq.write_table(table[start: end], chunk_filepath)

        print("Finished resharding the parquet files for optimized processing.")

        return new_items
