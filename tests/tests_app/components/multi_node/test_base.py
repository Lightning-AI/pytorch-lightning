from re import escape

import pytest
from tests_app.helpers.utils import no_warning_call

from lightning_app import CloudCompute, LightningWork
from lightning_app.components import MultiNode


def test_multi_node_warn_running_locally():
    class Work(LightningWork):
        def run(self):
            pass

    with pytest.warns(UserWarning, match=escape("You set MultiNode(num_nodes=1, ...)` but ")):
        MultiNode(Work, num_nodes=2, cloud_compute=CloudCompute("gpu"))

    with no_warning_call(UserWarning, match=escape("You set MultiNode(num_nodes=1, ...)` but ")):
        MultiNode(Work, num_nodes=1, cloud_compute=CloudCompute("gpu"))
