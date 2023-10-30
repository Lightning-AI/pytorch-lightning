import argparse
import sys
from multiprocessing import Process
from time import sleep
from unittest.mock import MagicMock

import pytest
import requests
from lightning.app import LightningApp, LightningFlow
from lightning.app.cli.commands.app_commands import _run_app_command
from lightning.app.cli.connect.app import connect_app, disconnect_app
from lightning.app.core.constants import APP_SERVER_PORT
from lightning.app.runners import MultiProcessRuntime
from lightning.app.testing.helpers import _RunIf
from lightning.app.utilities.commands.base import ClientCommand, _download_command, _validate_client_command
from lightning.app.utilities.state import AppState
from pydantic import BaseModel


class SweepConfig(BaseModel):
    sweep_name: str
    num_trials: int


class SweepCommand(ClientCommand):
    def run(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--sweep_name", type=str)
        parser.add_argument("--num_trials", type=int)
        hparams = parser.parse_args()

        config = SweepConfig(sweep_name=hparams.sweep_name, num_trials=hparams.num_trials)
        response = self.invoke_handler(config=config)
        assert response is True


class FlowCommands(LightningFlow):
    def __init__(self):
        super().__init__()
        self.names = []
        self.has_sweep = False

    def run(self):
        if self.has_sweep and len(self.names) == 1:
            sleep(1)
            self.stop()

    def trigger_method(self, name: str):
        print(name)
        self.names.append(name)

    def sweep(self, config: SweepConfig):
        self.has_sweep = True
        return True

    def configure_commands(self):
        return [{"user command": self.trigger_method}, {"sweep": SweepCommand(self.sweep)}]


class DummyConfig(BaseModel):
    something: str
    something_else: int


class DummyCommand(ClientCommand):
    def run(self, something: str, something_else: int) -> None:
        config = DummyConfig(something=something, something_else=something_else)
        response = self.invoke_handler(config=config)
        assert response == {"body": 0}


def run(config: DummyConfig):
    assert isinstance(config, DummyCommand)


def run_failure_0(name: str):
    pass


def run_failure_1(name):
    pass


class CustomModel(BaseModel):
    pass


def run_failure_2(name: CustomModel):
    pass


@_RunIf(skip_windows=True)
def test_validate_client_command():
    with pytest.raises(Exception, match="The provided annotation for the argument name"):
        _validate_client_command(ClientCommand(run_failure_0))

    with pytest.raises(Exception, match="annotate your method"):
        _validate_client_command(ClientCommand(run_failure_1))

    starts = "lightning.app".replace(".", "/")
    with pytest.raises(Exception, match=f"{starts}/utilities/commands/base.py"):
        _validate_client_command(ClientCommand(run_failure_2))


def test_client_commands(monkeypatch):
    import requests

    resp = MagicMock()
    resp.status_code = 200
    value = {"body": 0}
    resp.json = MagicMock(return_value=value)
    post = MagicMock()
    post.return_value = resp
    monkeypatch.setattr(requests, "post", post)
    url = "http//"
    kwargs = {"something": "1", "something_else": "1"}
    command = DummyCommand(run)
    _validate_client_command(command)
    client_command = _download_command(
        command_name="something",
        cls_path=__file__,
        cls_name="DummyCommand",
    )
    client_command._setup("something", app_url=url)
    client_command.run(**kwargs)


def target():
    app = LightningApp(FlowCommands())
    MultiProcessRuntime(app).dispatch()


@pytest.mark.xfail(strict=False, reason="failing for some reason, need to be fixed.")  # fixme
def test_configure_commands(monkeypatch):
    """This test validates command can be used locally with connect and disconnect."""
    process = Process(target=target)
    process.start()
    time_left = 15
    while time_left > 0:
        try:
            requests.get(f"http://localhost:{APP_SERVER_PORT}/healthz")
            break
        except requests.exceptions.ConnectionError:
            sleep(0.1)
            time_left -= 0.1

    sleep(0.5)
    monkeypatch.setattr(sys, "argv", ["lightning", "user", "command", "--name=something"])
    connect_app("localhost")
    _run_app_command("localhost", None)
    sleep(2)
    state = AppState()
    state._request_state()
    assert state.names == ["something"]
    monkeypatch.setattr(sys, "argv", ["lightning", "sweep", "--sweep_name=my_name", "--num_trials=1"])
    _run_app_command("localhost", None)
    time_left = 15
    while time_left > 0:
        if process.exitcode == 0:
            break
        sleep(0.1)
        time_left -= 0.1
    assert process.exitcode == 0
    disconnect_app()
    process.kill()
