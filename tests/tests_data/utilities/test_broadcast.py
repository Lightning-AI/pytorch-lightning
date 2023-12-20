import os
from unittest import mock

from lightning.data.utilities.broadcast import broadcast_object, requests


@mock.patch.dict(
    os.environ, {"LIGHTNING_APP_EXTERNAL_URL": "http://", "LIGHTNING_APP_STATE_URL": "http://"}, clear=True
)
def test_broadcast(monkeypatch):
    session = mock.MagicMock()
    resp = requests.Response()
    resp.status_code = 200

    def fn(*args, **kwargs):
        nonlocal session
        return {"value": session.post._mock_call_args_list[0].kwargs["json"]["value"]}

    resp.json = fn
    session.post.return_value = resp
    monkeypatch.setattr(requests, "Session", mock.MagicMock(return_value=session))
    assert broadcast_object("key", "value") == "value"
