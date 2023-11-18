import sys
from unittest import mock

import pytest
from lightning.app.cli.lightning_cli_delete import _find_selected_app_instance_id
from lightning_cloud.openapi import Externalv1LightningappInstance


@pytest.mark.skipif(sys.platform == "win32", reason="currently not supported for windows.")
@mock.patch("lightning_cloud.login.Auth.authenticate", mock.MagicMock())
@mock.patch("lightning.app.cli.lightning_cli_delete._AppManager.list_apps")
def test_app_find_selected_app_instance_id_when_app_name_exists(list_apps_mock: mock.MagicMock):
    list_apps_mock.return_value = [
        Externalv1LightningappInstance(name="app-name", id="app-id"),
    ]
    returned_app_instance_id = _find_selected_app_instance_id(app_name="app-name")
    assert returned_app_instance_id == "app-id"


@pytest.mark.skipif(sys.platform == "win32", reason="currently not supported for windows.")
@mock.patch("lightning_cloud.login.Auth.authenticate", mock.MagicMock())
@mock.patch("lightning.app.cli.lightning_cli_delete._AppManager.list_apps")
def test_app_find_selected_app_instance_id_when_app_id_exists(list_apps_mock: mock.MagicMock):
    list_apps_mock.return_value = [
        Externalv1LightningappInstance(name="app-name", id="app-id"),
    ]
    returned_app_instance_id = _find_selected_app_instance_id(app_name="app-id")
    assert returned_app_instance_id == "app-id"
