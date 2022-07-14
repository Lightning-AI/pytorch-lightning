from multiprocessing import Process
from time import sleep

from click.testing import CliRunner
from pydantic import BaseModel

from lightning import LightningFlow
from lightning_app import LightningApp
from lightning_app.cli.lightning_cli import command
from lightning_app.runners import MultiProcessRuntime
from lightning_app.utilities.commands import ClientCommand
from lightning_app.utilities.state import AppState


class SweepConfig(BaseModel):
    sweep_name: str
    num_trials: str


class SweepCommand(ClientCommand):
    def run(self, sweep_name: str, num_trials: str) -> None:
        config = SweepConfig(sweep_name=sweep_name, num_trials=num_trials)
        _ = self.invoke_handler(config=config)


class FlowCommands(LightningFlow):
    def __init__(self):
        super().__init__()
        self.names = []

    def trigger_method(self, name: str):
        self.names.append(name)

    def sweep(self, config: SweepConfig):
        print(config)

    def configure_commands(self):
        return [{"user_command": self.trigger_method}, {"sweep": SweepCommand(self.sweep)}]


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
