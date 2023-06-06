import os

import numpy as np
import pytest
from lightning_utilities.core.imports import package_available
from PIL import Image

from lightning.pytorch.utilities.data.get_index import get_index

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_test_index_data(index_path):
    with open(index_path) as f:
        data = f.readlines()
    return list(dict.fromkeys([item.split("/")[-1] for item in data if "jpeg" in item]))


@pytest.fixture(scope="session")
def image_set(tmp_path_factory):
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


@pytest.mark.skipif(not package_available("lightning"), reason="Supported only with mono-package")
def test_get_index_generate_for_s3_bucket():
    """Can generate an index as s3 bucket mounted localled on the Lightning AI platform."""
    test_index_path = f"{THIS_DIR}/test_data/test_index_s3.txt"
    test_index_data = get_test_index_data(test_index_path)

    test_bucket = "s3://nohaspublictestbucket"
    index_path = os.path.join(os.getcwd(), "index_1.txt")
    got_index = get_index(s3_connection_path=test_bucket, index_file_path=index_path)

    assert got_index

    generated_index = get_test_index_data(index_path)

    assert len(test_index_data) == len(generated_index)
    assert test_index_data == generated_index


@pytest.mark.skipif(not package_available("lightning"), reason="Supported only with mono-package")
def test_get_index_generate_for_local_folder(image_set):
    """Can generate an index for an s3 bucket."""
    test_index_path = f"{THIS_DIR}/test_data/test_index.txt"
    test_index_data = get_test_index_data(test_index_path)

    # test_local_bucket = "data/test_dataset"
    index_path = os.path.join(THIS_DIR, "index_2.txt")
    got_index = get_index(s3_connection_path=image_set, index_file_path=index_path)

    assert got_index

    generated_index = get_test_index_data(index_path)

    assert len(test_index_data) == len(generated_index)

    item_from_gen_list = list(dict.fromkeys([item.split("/")[-1] for item in generated_index if "jpeg" in item]))
    assert sorted(test_index_data) == sorted(item_from_gen_list)


@pytest.mark.skip(reason="Not required at the moment")
def test_get_index_generate_for_mounted_s3_bucket():
    """Can generate an index for an s3 bucket."""
    test_index_path = f"{THIS_DIR}/test_data/test_index_s3.txt"
    test_index_data = get_test_index_data(test_index_path)

    test_local_bucket = "/data/nohaspublictestbucket"
    index_path = os.path.join(THIS_DIR, "index_3.txt")
    got_index = get_index(s3_connection_path=test_local_bucket, index_file_path=index_path)

    assert got_index

    generated_index = get_test_index_data(index_path)

    assert len(test_index_data) == len(generated_index)
    assert test_index_data == generated_index
