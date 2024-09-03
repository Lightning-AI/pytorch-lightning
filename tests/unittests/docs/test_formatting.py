import os.path
import re

import pytest

from lightning_utilities.docs import adjust_linked_external_docs


@pytest.mark.online
def test_adjust_linked_external_docs(temp_docs):
    # take config as it includes API references with `stable`
    path_conf = os.path.join(temp_docs, "conf.py")
    path_testpage = os.path.join(temp_docs, "test-page.rst")

    def _get_line_with_numpy(path_rst: str, pattern: str) -> str:
        with open(path_rst, encoding="UTF-8") as fopen:
            lines = fopen.readlines()
        # find the first line with figure reference
        return next(ln for ln in lines if pattern in ln)

    # validate the initial expectations
    line = _get_line_with_numpy(path_conf, pattern='"numpy":')
    assert "https://numpy.org/doc/stable/" in line
    line = _get_line_with_numpy(path_testpage, pattern="Link to scikit-learn stable documentation:")
    assert "https://scikit-learn.org/stable/" in line

    adjust_linked_external_docs(
        "https://numpy.org/doc/stable/", "https://numpy.org/doc/{numpy.__version__}/", temp_docs
    )
    adjust_linked_external_docs(
        "https://scikit-learn.org/stable/", "https://scikit-learn.org/{scikit-learn}/", temp_docs
    )

    # validate the final state of index page
    line = _get_line_with_numpy(path_conf, pattern='"numpy":')
    assert re.search(r"https://numpy.org/doc/([1-9]\d*)\.(\d+)/", line)
    line = _get_line_with_numpy(path_testpage, pattern="Link to scikit-learn stable documentation:")
    assert re.search(r"https://scikit-learn.org/([1-9]\d*)\.(\d+)/", line)
