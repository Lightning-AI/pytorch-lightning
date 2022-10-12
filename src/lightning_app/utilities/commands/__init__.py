from lightning_app.utilities.commands.base import _commands_to_api, ClientCommand


def get_default_commands():
    from lightning_app.utilities.commands.artifacts.download import DOWNLOAD_ARTIFACT
    from lightning_app.utilities.commands.artifacts.show import SHOW_ARTIFACT
    from lightning_app.utilities.commands.artifacts.upload import UPLOAD_ARTIFACT

    return _commands_to_api([DOWNLOAD_ARTIFACT, SHOW_ARTIFACT, UPLOAD_ARTIFACT])


__all__ = ["ClientCommand", "get_default_commands"]
