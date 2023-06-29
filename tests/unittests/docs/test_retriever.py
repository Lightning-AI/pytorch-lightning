import os.path
import shutil

from unittests import _PATH_ROOT

from lightning_utilities.docs import fetch_external_assets


def test_retriever_s3():
    path_docs = os.path.join(_PATH_ROOT, "docs", "source")
    path_index = os.path.join(path_docs, "index.rst")
    path_page = os.path.join(path_docs, "any", "extra", "page.rst")
    os.makedirs(os.path.dirname(path_page), exist_ok=True)
    shutil.copy(path_index, path_page)

    fetch_external_assets(docs_folder=path_docs)

    with open(path_index, encoding="UTF-8") as fo:
        body = fo.read()
    # that the image exists~
    assert "Lightning.gif" in body
    # but it is not sourced from S3
    assert ".s3." not in body

    with open(path_page, encoding="UTF-8") as fo:
        body = fo.read()
    # that the image exists~
    assert "Lightning.gif" in body
    # check the proper depth
    assert os.path.sep.join(["..", ".."]) in body
