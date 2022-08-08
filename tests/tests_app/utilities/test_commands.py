import argparse
import sys
from multiprocessing import Process
from time import sleep
from unittest.mock import MagicMock

import pytest
import requests
from pydantic import BaseModel

from lightning import LightningFlow
from lightning_app import LightningApp
from lightning_app.cli.lightning_cli import app_command
from lightning_app.core.constants import APP_SERVER_PORT
from lightning_app.runners import MultiProcessRuntime
from lightning_app.testing.helpers import RunIf
from lightning_app.utilities.commands.base import _command_to_method_and_metadata, _download_command, ClientCommand
from lightning_app.utilities.state import AppState


class SweepConfig(BaseModel):
    sweep_name: str
    num_trials: int


class SweepCommand(ClientCommand):
    def run(self) -> None:
        print(sys.argv)
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
            self._exit()

    def trigger_method(self, name: str):
        self.names.append(name)

    def sweep(self, config: SweepConfig):
        self.has_sweep = True
        return True

    def configure_commands(self):
        return [{"user_command": self.trigger_method}, {"sweep": SweepCommand(self.sweep)}]


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


@RunIf(skip_windows=True)
def test_command_to_method_and_metadata():
    with pytest.raises(Exception, match="The provided annotation for the argument name"):
        _command_to_method_and_metadata(ClientCommand(run_failure_0))

    with pytest.raises(Exception, match="annotate your method"):
        _command_to_method_and_metadata(ClientCommand(run_failure_1))

    with pytest.raises(Exception, match="lightning_app/utilities/commands/base.py"):
        _command_to_method_and_metadata(ClientCommand(run_failure_2))


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
    _, command_metadata = _command_to_method_and_metadata(command)
    command_metadata.update(
        {
            "command": "dummy",
            "affiliation": "root",
            "is_client_command": True,
            "owner": "root",
        }
    )
    client_command, models = _download_command(command_metadata, None)
    client_command._setup(metadata=command_metadata, models=models, app_url=url)
    client_command.run(**kwargs)


def target():
    app = LightningApp(FlowCommands())
    MultiProcessRuntime(app).dispatch()


def test_configure_commands(monkeypatch):
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
    monkeypatch.setattr(sys, "argv", ["lightning", "user_command", "--name=something"])
    app_command()
    sleep(0.5)
    state = AppState()
    state._request_state()
    assert state.names == ["something"]
    monkeypatch.setattr(sys, "argv", ["lightning", "sweep", "--sweep_name", "my_name", "--num_trials", "1"])
    app_command()
    time_left = 15
    while time_left > 0 and process.exitcode != 0:
        sleep(0.1)
        time_left -= 0.1
    assert process.exitcode == 0
