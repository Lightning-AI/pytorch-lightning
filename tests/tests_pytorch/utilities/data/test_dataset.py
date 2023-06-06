import os
import socket

import pytest

from lightning.pytorch.utilities.data.dataset import LightningDataset
from lightning.pytorch.utilities.data.fileio import OpenCloudFileObj
from lightning_utilities import module_available
from lightning_utilities.core.imports import package_available


def isConnectedWithInternet():
    try:
        socket.create_connection(("1.1.1.1", 53))
        return True
    except OSError:
        pass
    return False


@pytest.mark.skipif(
    not isConnectedWithInternet(),
    reason="Not connected to internet"
)
@pytest.mark.skipif(
    not package_available('lightning'),
    reason="Supported only with mono-package"
)
def test_lightning_dataset(tmpdir):
    index_path = os.path.join(tmpdir, "index.txt")
    # TODO: adapt this once the fallback and tests for get_index are ready!
    dset = LightningDataset("s3://imagenet-resized", path_to_index_file=index_path)
    tuple_of_files = dset.get_index()
    assert isinstance(tuple_of_files, tuple)
    assert all(map(lambda x: isinstance(x, str)))

    assert os.path.isfile(index_path)
    with open(index_path) as f:
        file_content = f.readlines()

    assert len(file_content) == len(tuple_of_files)

    for file_entry, tuple_entry in zip(file_content, tuple_of_files):
        assert file_content == tuple_entry + "\n"

    assert isinstance(dset.open(index_path), OpenCloudFileObj)

    foo_path = os.path.join(tmpdir, "foo.txt")
    with open(foo_path, "w") as f:
        f.write("bar!")

    with dset.open(foo_path, "r") as f:
        assert f.read() == "bar!"

    with dset.open(foo_path, "w") as f:
        f.write("not bar anymore!")

    with open(foo_path) as f:
        assert f.read() == "not bar anymore!"

    with open(foo_path, "w") as f:
        f.write("bar again!")

    file_obj = dset.open(foo_path, "w")
    assert file_obj.read() == "bar again!"
    file_obj.close()
    assert file_obj._stream.closed
