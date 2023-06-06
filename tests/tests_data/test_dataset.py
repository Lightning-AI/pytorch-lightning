import os
import socket
from types import GeneratorType

import numpy as np
import pytest
from lightning_utilities.core.imports import package_available

from lightning.data.dataset import LightningDataset
from lightning.data.fileio import OpenCloudFileObj


def isConnectedWithInternet():
    try:
        socket.create_connection(("1.1.1.1", 53))
        return True
    except OSError:
        pass
    return False


@pytest.fixture(scope="session")
def image_set(tmp_path_factory):
    from PIL import Image

    file_nums = [
        0,
        1,
        10,
        100,
        1000,
        10000,
        100000,
        1000000,
        1000001,
        1000002,
        1000003,
        1000004,
        1000005,
        1000006,
        1000007,
        1000008,
        1000009,
    ]

    img = np.random.randint(255, size=(800, 800))
    img = img.astype(np.uint8)
    im = Image.fromarray(img)

    for i in file_nums:
        fn = tmp_path_factory.mktemp("test_data") / f"img-{i}.jpeg"
        im.save(fn)

    return tmp_path_factory.getbasetemp()._str


@pytest.mark.skipif(not isConnectedWithInternet(), reason="Not connected to internet")
@pytest.mark.skipif(not package_available("lightning"), reason="Supported only with mono-package")
def test_lightning_dataset(tmpdir, image_set):
    index_path = os.path.join(tmpdir, "index.txt")
    # TODO: adapt this once the fallback and tests for get_index are ready!
    dset = LightningDataset(image_set, path_to_index_file=index_path)
    tuple_of_files = dset.get_index()
    assert isinstance(tuple_of_files, GeneratorType)
    files_list = list(tuple_of_files)

    assert os.path.isfile(index_path)
    with open(index_path) as f:
        file_content = f.readlines()

    assert len(file_content) == len(files_list)

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

    file_obj = dset.open(foo_path, "w")
    file_obj.close()
    assert file_obj._stream.closed
