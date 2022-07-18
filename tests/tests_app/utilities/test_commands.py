from multiprocessing import Process
from time import sleep
from unittest.mock import MagicMock

from click.testing import CliRunner
from pydantic import BaseModel

from lightning import LightningFlow
from lightning_app import LightningApp
from lightning_app.cli.lightning_cli import command
from lightning_app.runners import MultiProcessRuntime
from lightning_app.utilities.commands.base import _command_to_method_and_metadata, _download_command, ClientCommand
from lightning_app.utilities.state import AppState


class SweepConfig(BaseModel):
    sweep_name: str
    num_trials: str


class SweepCommand(ClientCommand):
    def run(self, sweep_name: str, num_trials: str) -> None:
        config = SweepConfig(sweep_name=sweep_name, num_trials=num_trials)
        response = self.invoke_handler(config=config)
        assert response is True


class FlowCommands(LightningFlow):
    def __init__(self):
        super().__init__()
        self.names = []

    def trigger_method(self, name: str):
        self.names.append(name)

    def sweep(self, config: SweepConfig):
        print(config)
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
            "is_command": True,
            "owner": "root",
        }
    )
    client_command, models = _download_command(command_metadata)
    client_command._setup(metadata=command_metadata, models=models, url=url)
    client_command.run(**kwargs)


def target():
    app = LightningApp(FlowCommands())
    MultiProcessRuntime(app).dispatch()


def test_configure_commands():
    process = Process(target=target)
    process.start()
    sleep(5)
    runner = CliRunner()
    result = runner.invoke(
        command,
        ["user_command", "--args", "name=something"],
        catch_exceptions=False,
    )
    sleep(2)
    assert result.exit_code == 0
    state = AppState()
    state._request_state()
    assert state.names == ["something"]
    runner = CliRunner()
    result = runner.invoke(
        command,
        ["sweep", "--args", "sweep_name=my_name", "--args", "num_trials=num_trials"],
        catch_exceptions=False,
    )
    sleep(2)
    assert result.exit_code == 0
