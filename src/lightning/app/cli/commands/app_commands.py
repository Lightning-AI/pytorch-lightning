# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from typing import Dict, Optional

import requests

from lightning.app.cli.connect.app import (
    _clean_lightning_connection,
    _install_missing_requirements,
    _resolve_command_path,
)
from lightning.app.utilities.cli_helpers import _LightningAppOpenAPIRetriever
from lightning.app.utilities.commands.base import _download_command
from lightning.app.utilities.enum import OpenAPITags


def _is_running_help(argv) -> bool:
    return argv[-1] in ["--help", "-"] if argv else False


def _run_app_command(app_name: str, app_id: Optional[str]):
    """Execute a function in a running App from its name."""
    # 1: Collect the url and comments from the running application
    _clean_lightning_connection()

    running_help = _is_running_help(sys.argv)

    retriever = _LightningAppOpenAPIRetriever(app_id, use_cache=running_help)

    if not running_help and (retriever.url is None or retriever.api_commands is None):
        if app_name == "localhost":
            print("The command couldn't be executed as your local Lightning App isn't running.")
        else:
            print(f"The command couldn't be executed as your cloud Lightning App `{app_name}` isn't running.")
        sys.exit(0)

    if not retriever.api_commands:
        raise Exception("This application doesn't expose any commands yet.")

    full_command = "_".join(sys.argv)

    has_found = False
    for command in list(retriever.api_commands):
        if command in full_command:
            has_found = True
            for value in sys.argv:
                if value == command and "_" in value:
                    print(
                        f"The command `{value}` was provided with an underscore and it isn't allowed."
                        f"Instead, use `lightning {value.replace('_', ' ')}`."
                    )
                    sys.exit(0)
            break

    if not has_found:
        raise Exception(f"The provided command isn't available in {list(retriever.api_commands)}")

    # 2: Send the command from the user
    metadata = retriever.api_commands[command]

    try:
        # 3: Execute the command
        if metadata["tag"] == OpenAPITags.APP_COMMAND:
            _handle_command_without_client(command, metadata, retriever.url)
        else:
            _handle_command_with_client(command, metadata, app_name, app_id, retriever.url)
    except ModuleNotFoundError:
        _install_missing_requirements(retriever, fail_if_missing=True)

    if running_help:
        print("Your command execution was successful.")


def _handle_command_without_client(command: str, metadata: Dict, url: str) -> None:
    supported_params = list(metadata["parameters"])
    if _is_running_help(sys.argv):
        print(f"Usage: lightning {command} [ARGS]...")
        print(" ")
        print("Options")
        for param in supported_params:
            print(f"  {param}: Add description")
        return

    provided_params = [param.replace("--", "") for param in sys.argv[1 + len(command.split("_")) :]]

    # TODO: Add support for more argument types.
    if any("=" not in param for param in provided_params):
        raise Exception("Please, use --x=y syntax when providing the command arguments.")

    if any(param.split("=")[0] not in supported_params for param in provided_params):
        raise Exception(f"Some arguments need to be provided. The keys are {supported_params}.")

    # TODO: Encode the parameters and validate their type.
    query_parameters = "&".join(provided_params)
    resp = requests.post(url + f"/command/{command}?{query_parameters}")
    assert resp.status_code == 200, resp.json()
    print(resp.json())


def _handle_command_with_client(command: str, metadata: Dict, app_name: str, app_id: Optional[str], url: str):
    debug_mode = bool(int(os.getenv("DEBUG", "0")))

    if app_name == "localhost":
        target_file = metadata["cls_path"]
    else:
        target_file = _resolve_command_path(command) if debug_mode else _resolve_command_path(command)

    if debug_mode:
        print(target_file)

    client_command = _download_command(
        command,
        metadata["cls_path"],
        metadata["cls_name"],
        app_id,
        debug_mode=debug_mode,
        target_file=target_file if debug_mode else _resolve_command_path(command),
    )
    client_command._setup(command_name=command, app_url=url)
    sys.argv = sys.argv[len(command.split("_")) :]
    client_command.run()
