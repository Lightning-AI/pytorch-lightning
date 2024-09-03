import os.path
import shutil

import pytest

from lightning_utilities.docs import fetch_external_assets


@pytest.mark.online
def test_retriever_s3(temp_docs):
    # take the index page
    path_index = os.path.join(temp_docs, "index.rst")
    # copy it to another location to test depth
    path_page = os.path.join(temp_docs, "any", "extra", "page.rst")
    os.makedirs(os.path.dirname(path_page), exist_ok=True)
    shutil.copy(path_index, path_page)

    def _get_line_with_figure(path_rst: str) -> str:
        with open(path_rst, encoding="UTF-8") as fopen:
            lines = fopen.readlines()
        # find the first line with figure reference
        return next(ln for ln in lines if ln.startswith(".. figure::"))

    # validate the initial expectations
    line = _get_line_with_figure(path_index)
    # that the image exists
    assert "Lightning.gif" in line
    # and it is sourced in S3
    assert ".s3." in line

    fetch_external_assets(docs_folder=temp_docs)

    # validate the final state of index page
    line = _get_line_with_figure(path_index)
    # that the image exists
    assert os.path.join("fetched-s3-assets", "Lightning.gif") in line
    # but it is not sourced from S3
    assert ".s3." not in line

    # validate the final state of additional page
    line = _get_line_with_figure(path_page)
    # that the image exists in the proper depth
    assert os.path.join("..", "..", "fetched-s3-assets", "Lightning.gif") in line
    # but it is not sourced from S3
    assert ".s3." not in line
