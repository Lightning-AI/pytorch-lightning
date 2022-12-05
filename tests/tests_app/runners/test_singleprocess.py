import os
from unittest import mock

import pytest

from lightning_app import LightningFlow
from lightning_app.core.app import LightningApp
from lightning_app.runners import SingleProcessRuntime


class Flow(LightningFlow):
    def run(self):
        raise KeyboardInterrupt


def on_before_run():
    pass


def test_single_process_runtime(tmpdir):

    app = LightningApp(Flow())
    SingleProcessRuntime(app, start_server=False).dispatch(on_before_run=on_before_run)


@pytest.mark.parametrize(
    "env,expected_url",
    [
        ({}, "http://127.0.0.1:7501/view"),
        ({"APP_SERVER_HOST": "http://test"}, "http://test"),
    ],
)
def test_get_app_url(env, expected_url):
    with mock.patch.dict(os.environ, env):
        assert SingleProcessRuntime._get_app_url() == expected_url
