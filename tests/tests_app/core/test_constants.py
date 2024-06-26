import json
import os
from unittest import mock

from lightning.app.core.constants import DistributedPluginChecker, get_lightning_cloud_url


@mock.patch.dict(os.environ, {"LIGHTNING_CLOUD_URL": "https://beta.lightning.ai"})
def test_defaults():
    assert get_lightning_cloud_url() == "https://beta.lightning.ai"


def test_distributed_checker(monkeypatch):
    monkeypatch.setenv("DISTRIBUTED_ARGUMENTS", str(json.dumps({"num_instances": 2})))
    monkeypatch.setenv("LIGHTNING_CLOUD_WORK_NAME", "nodes.0")
    assert bool(DistributedPluginChecker())

    monkeypatch.setenv("LIGHTNING_CLOUD_WORK_NAME", "nodes.1")
    assert bool(DistributedPluginChecker())

    monkeypatch.setenv("LIGHTNING_CLOUD_WORK_NAME", "nodes.2")
    assert bool(DistributedPluginChecker())

    mock_work = mock.MagicMock()
    mock_work.name = "nodes.1"
    assert not DistributedPluginChecker().should_create_work(mock_work)

    mock_work.name = "nodes.2"
    assert DistributedPluginChecker().should_create_work(mock_work)
