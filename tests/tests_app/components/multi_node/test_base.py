from re import escape
from unittest import mock

import pytest
from lightning.app import CloudCompute, LightningWork
from lightning.app.components import MultiNode
from lightning_utilities.test.warning import no_warning_call


def test_multi_node_warn_running_locally():
    class Work(LightningWork):
        def run(self):
            pass

    with pytest.warns(UserWarning, match=escape("You set MultiNode(num_nodes=2, ...)` but ")):
        MultiNode(Work, num_nodes=2, cloud_compute=CloudCompute("gpu"))

    with no_warning_call(UserWarning, match=escape("You set MultiNode(num_nodes=1, ...)` but ")):
        MultiNode(Work, num_nodes=1, cloud_compute=CloudCompute("gpu"))


@mock.patch("lightning.app.components.multi_node.base.is_running_in_cloud", mock.Mock(return_value=True))
def test_multi_node_separate_cloud_computes():
    class Work(LightningWork):
        def run(self):
            pass

    m = MultiNode(Work, num_nodes=2, cloud_compute=CloudCompute("gpu"))

    assert len({w.cloud_compute._internal_id for w in m.ws}) == len(m.ws)
