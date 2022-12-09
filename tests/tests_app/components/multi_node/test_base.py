from re import escape
from unittest import mock

import pytest
from tests_app.helpers.utils import no_warning_call

import lightning_app
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


@mock.patch("lightning_app.components.multi_node.base.is_running_in_cloud", mock.Mock(return_value=True))
def test_multi_node_separate_cloud_computes():
    class Work(LightningWork):
        def run(self):
            pass

    MultiNode(Work, num_nodes=2, cloud_compute=CloudCompute("gpu"))

    assert len(lightning_app.utilities.packaging.cloud_compute._CLOUD_COMPUTE_STORE) == 2
    for v in lightning_app.utilities.packaging.cloud_compute._CLOUD_COMPUTE_STORE.values():
        assert len(v.component_names) == 1
        assert v.component_names[0].startswith("root.ws.") and v.component_names[0][-1].isdigit()
