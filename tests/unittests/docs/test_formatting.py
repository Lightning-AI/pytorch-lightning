import os.path

import pytest
from lightning_utilities.docs import adjust_linked_external_docs


@pytest.mark.online()
def test_adjust_linked_external_docs(temp_docs):
    # take config as it includes API references with `stable`
    path_conf = os.path.join(temp_docs, "conf.py")

    def _get_line_with_numpy(path_rst: str) -> str:
        with open(path_rst, encoding="UTF-8") as fopen:
            lines = fopen.readlines()
        # find the first line with figure reference
        return next(ln for ln in lines if ln.lstrip().startswith('"numpy":'))

    # validate the initial expectations
    line = _get_line_with_numpy(path_conf)
    assert "https://numpy.org/doc/stable/" in line

    adjust_linked_external_docs(
        "https://numpy.org/doc/stable/", "https://numpy.org/doc/{numpy.__version__}/", temp_docs
    )

    import numpy as np

    np_version = np.__version__.split(".")

    # validate the final state of index page
    line = _get_line_with_numpy(path_conf)
    assert f"https://numpy.org/doc/{'.'.join(np_version[:2])}/" in line
