import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def run_app_commands(file: str):
    command_lines = []
    command_line_numbers = []
    with open(file) as f:
        app_lines = f.readlines()

    for line_number, line in enumerate(app_lines):
        # only execute comment lines
        if not line.startswith("#"):
            break

        # remove comment marker and any leading / trailing whitespaces
        line = line.lstrip("#").strip()
        if len(line) == 0:
            # do not block on comment lines
            continue

        # only run commands starting with a bang (!)
        if line[0] != "!":
            continue

        command_lines.append(line)
        command_line_numbers.append(line_number)

    if len(command_lines) == 0:
        logger.debug("No in app commands to install.")
        return

    logger.debug(f"Found app commands to install, running: {command_lines}")

    for command, line_number in zip(command_lines, command_line_numbers):
        completed = subprocess.run(
            command,
            shell=True,
            env=os.environ,
        )
        try:
            completed.check_returncode()
        except subprocess.CalledProcessError as e:
            logger.error(
                f"There was a problem on line {line_number} of {file} while executing the command {command}."
                f"Please check the command and try executing this again. "
            )
            raise e
