import os
import sys

import pytest
from lightning.data import map
from lightning.data.processing.readers import _POLARS_AVAILABLE, _PYARROW_AVAILABLE, BaseReader, ParquetReader


class DummyReader(BaseReader):

    def items_to_workers(self, items, num_workers: int):
        return [[(worker_idx, idx, item) for idx, item in enumerate(items)] for worker_idx in range(num_workers)]

    def read(self, item):
        return item


def fn(data: str, output_dir):
    worker_idx, idx, _ = data

    with open(os.path.join(output_dir, f"{worker_idx}_{idx}"), "w") as f:
        f.write("hello world")


def test_reader(tmpdir):
    map(fn, list(range(3)), output_dir=str(tmpdir), reader=DummyReader(), num_workers=2)
    assert sorted(os.listdir(tmpdir)) == ['0_0', '0_1', '0_2', '1_0', '1_1', '1_2']


def map_parquet(df, output_dir):
    filename = f"{df.row(0)[0]}_{len(df)}"

    with open(os.path.join(output_dir, filename), "w") as f:
        f.write("hello world")

@pytest.mark.skipif(
    (not _POLARS_AVAILABLE and not _PYARROW_AVAILABLE) or sys.platform == "linux",
    reason="polars and pyarrow are required"
)
def test_parquet_reader(tmpdir):
    import polars as pol

    inputs = []

    for i in range(3):
        parquet_path = os.path.join(tmpdir, f"{i}.parquet")
        df = pol.DataFrame(list(range(i * 10, (i + 1) * 10)))
        df.write_parquet(parquet_path)
        inputs.append(parquet_path)

    map(
        map_parquet,
        inputs=inputs,
        output_dir=os.path.join(tmpdir, "output_dir"),
        reader=ParquetReader(num_rows=10, to_pandas=False),
        num_workers=2
    )

    assert sorted(os.listdir(os.path.join(tmpdir, "output_dir"))) == ['0_10', '10_5', '15_5', '20_10']
