from lightning_app.utilities.commands.artifacts.show import SHOW_ARTIFACT
from lightning_app.utilities.commands.base import _commands_to_api, ClientCommand


def get_default_commands():
    return _commands_to_api(
        [
            SHOW_ARTIFACT,
        ]
    )


__all__ = ["ClientCommand", "get_default_commands"]
