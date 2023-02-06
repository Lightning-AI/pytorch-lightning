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

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import List

from lightning_app.utilities.exceptions import MisconfigurationException

logger = logging.getLogger(__name__)

# These are common lines at the top of python files which conflict with our
# command syntax but which should not be executed. This is non-exhaustive,
# and it may be better to just ignoring shebang lines if we see problems here.
APP_COMMAND_LINES_TO_IGNORE = {
    "#!/usr/bin/python",
    "#!/usr/local/bin/python",
    "#!/usr/bin/env python",
    "#!/usr/bin/env python3",
}


@dataclass
class CommandLines:
    file: str
    commands: List[str] = field(default_factory=list)
    line_numbers: List[int] = field(default_factory=list)


def _extract_commands_from_file(file_name: str) -> CommandLines:
    """Extract all lines at the top of the file which contain commands to execute.

    The return struct contains a list of commands to execute with the corresponding line number the command executed on.
    """
    cl = CommandLines(
        file=file_name,
    )
    with open(file_name) as f:
        file_lines = f.readlines()

    for line_number, line in enumerate(file_lines):
        line = line.strip()
        if line in APP_COMMAND_LINES_TO_IGNORE:
            continue

        # stop parsing at first non-comment line at top of file
        if not line.startswith("#"):
            continue

        # remove comment marker and any leading / trailing whitespaces
        line = line.lstrip("#").strip()
        if len(line) == 0:
            # do not stop parsing on empty on comment lines
            continue

        # only run commands starting with a bang (!) & strip the bang from the executed command.
        if line[0] != "!":
            continue
        line = line[1:].strip()

        cl.commands.append(line)
        # add 1 to line number because enumerate returns indexes starting at 0, while
        # text exitors list lines beginning at index 1.
        cl.line_numbers.append(line_number + 1)

    return cl


def _execute_app_commands(cl: CommandLines) -> None:
    """open a subprocess shell to execute app commands.

    The calling app environment is used in the current environment the code is running in
    """
    for command, line_number in zip(cl.commands, cl.line_numbers):
        logger.info(f"Running app setup command: {command}")
        completed = subprocess.run(
            command,
            shell=True,
            env=os.environ,
        )
        try:
            completed.check_returncode()
        except subprocess.CalledProcessError:
            err_txt = (
                f"There was a problem on line {line_number} of {cl.file} while executing the command: "
                f"{command}. More information on the problem is shown in the output above this "
                f"message. After editing this line to fix the problem you can run the app again."
            )
            logger.error(err_txt)
            raise MisconfigurationException(err_txt) from None


def run_app_commands(file: str) -> None:
    """Extract all lines at the top of the file which contain commands & execute them.

    Commands to execute are comment lines whose first non-whitespace character
    begins with the "bang" symbol (`!`).  After the first non comment line we
    stop parsing the rest of the file. Running environment is preserved in the
    subprocess shell.

    For example:

        # some file           <--- not a command
        # !echo "hello world" <--- a command
        # ! pip install foo   <--- a command
        # foo! bar            <--- not a command
        import lightning      <--- not a command, end parsing.

        where `echo "hello world" && pip install foo` would be executed in the current
        running environment.
    """
    cl = _extract_commands_from_file(file_name=file)
    if len(cl.commands) == 0:
        logger.debug("No in app commands to install.")
        return
    _execute_app_commands(cl=cl)
