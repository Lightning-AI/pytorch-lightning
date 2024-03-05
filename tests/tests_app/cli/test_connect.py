import json
import os
from unittest.mock import MagicMock

import click
import psutil
import pytest
from lightning.app import _PROJECT_ROOT
from lightning.app.cli.connect.app import (
    _list_app_commands,
    _resolve_command_path,
    _retrieve_connection_to_an_app,
    connect_app,
    disconnect_app,
)
from lightning.app.utilities import cli_helpers
from lightning.app.utilities.commands import base


def monkeypatch_connection(monkeypatch, tmpdir, ppid):
    connection_path = os.path.join(tmpdir, ppid)
    monkeypatch.setattr("lightning.app.cli.connect.app._clean_lightning_connection", MagicMock())
    monkeypatch.setattr("lightning.app.cli.connect.app._PPID", ppid)
    monkeypatch.setattr("lightning.app.cli.connect.app._LIGHTNING_CONNECTION", tmpdir)
    monkeypatch.setattr("lightning.app.cli.connect.app._LIGHTNING_CONNECTION_FOLDER", connection_path)
    return connection_path


@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_connect_disconnect_local(tmpdir, monkeypatch):
    disconnect_app()

    with pytest.raises(Exception, match="Connection wasn't successful. Is your app localhost running ?"):
        connect_app("localhost")

    with open(os.path.join(os.path.dirname(__file__), "jsons/connect_1.json")) as f:
        data = json.load(f)

    data["paths"]["/command/command_with_client"]["post"]["cls_path"] = os.path.join(
        _PROJECT_ROOT,
        data["paths"]["/command/command_with_client"]["post"]["cls_path"],
    )

    messages = []

    disconnect_app()

    def fn(msg):
        messages.append(msg)

    monkeypatch.setattr(click, "echo", fn)

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = data
    monkeypatch.setattr(cli_helpers.requests, "get", MagicMock(return_value=response))
    connect_app("localhost")
    assert _retrieve_connection_to_an_app() == ("localhost", None)
    command_path = _resolve_command_path("nested_command")
    assert not os.path.exists(command_path)
    command_path = _resolve_command_path("command_with_client")
    assert os.path.exists(command_path)
    messages = []
    connect_app("localhost")
    assert messages == ["You are connected to the local Lightning App."]

    messages = []
    disconnect_app()
    assert messages == ["You are disconnected from the local Lightning App."]
    messages = []
    disconnect_app()
    assert messages == [
        "You aren't connected to any Lightning App."
        " Please use `lightning_app connect app_name_or_id` to connect to one."
    ]

    assert _retrieve_connection_to_an_app() == (None, None)


def test_connect_disconnect_cloud(tmpdir, monkeypatch):
    disconnect_app()

    ppid_1 = str(psutil.Process(os.getpid()).ppid())
    ppid_2 = "222"

    target_file = _resolve_command_path("command_with_client")

    if os.path.exists(target_file):
        os.remove(target_file)

    with open(os.path.join(os.path.dirname(__file__), "jsons/connect_1.json")) as f:
        data = json.load(f)

    data["paths"]["/command/command_with_client"]["post"]["cls_path"] = os.path.join(
        _PROJECT_ROOT,
        data["paths"]["/command/command_with_client"]["post"]["cls_path"],
    )

    messages = []

    def fn(msg):
        messages.append(msg)

    monkeypatch.setattr(click, "echo", fn)

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = data
    monkeypatch.setattr(cli_helpers.requests, "get", MagicMock(return_value=response))
    project = MagicMock()
    project.project_id = "custom_project_name"
    monkeypatch.setattr(cli_helpers, "_get_project", MagicMock(return_value=project))
    client = MagicMock()
    lightningapps = MagicMock()

    app = MagicMock()
    app.display_name = "example"
    app.id = "1234"

    lightningapps.lightningapps = [app]
    client.lightningapp_instance_service_list_lightningapp_instances.return_value = lightningapps
    monkeypatch.setattr(cli_helpers, "LightningClient", MagicMock(return_value=client))

    monkeypatch.setattr(base, "_get_project", MagicMock(return_value=project))

    artifact = MagicMock()
    artifact.filename = "commands/command_with_client.py"
    artifacts = MagicMock()
    artifacts.artifacts = [artifact]
    client.lightningapp_instance_service_list_lightningapp_instance_artifacts.return_value = artifacts
    monkeypatch.setattr(base, "LightningClient", MagicMock(return_value=client))

    with open(data["paths"]["/command/command_with_client"]["post"]["cls_path"], "rb") as f:
        response.content = f.read()

    connect_app("example")
    assert _retrieve_connection_to_an_app() == ("example", "1234")
    commands = _list_app_commands()
    assert commands == ["command with client", "command without client", "nested command"]
    command_path = _resolve_command_path("nested_command")
    assert not os.path.exists(command_path)
    command_path = _resolve_command_path("command_with_client")
    assert os.path.exists(command_path)
    messages = []
    connect_app("example")
    assert messages == ["You are already connected to the cloud Lightning App: example."]

    _ = monkeypatch_connection(monkeypatch, tmpdir, ppid=ppid_2)

    messages = []
    connect_app("example")
    assert "The lightning App CLI now responds to app commands" in messages[0]

    messages = []
    disconnect_app()
    assert messages == ["You are disconnected from the cloud Lightning App: example."]

    _ = monkeypatch_connection(monkeypatch, tmpdir, ppid=ppid_1)

    messages = []
    disconnect_app()
    assert "You aren't connected to any Lightning App" in messages[0]

    messages = []
    disconnect_app()
    assert messages == [
        "You aren't connected to any Lightning App."
        " Please use `lightning_app connect app_name_or_id` to connect to one."
    ]

    assert _retrieve_connection_to_an_app() == (None, None)
