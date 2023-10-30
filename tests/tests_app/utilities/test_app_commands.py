import os
import sys

import pytest
from lightning.app.utilities.app_commands import CommandLines, _execute_app_commands, _extract_commands_from_file
from lightning.app.utilities.exceptions import MisconfigurationException


@pytest.mark.parametrize(
    ("filename", "expected_commands", "expected_line_numbers"),
    [
        ("single_command.txt", ['echo "foo"'], [1]),
        ("multiple_commands.txt", ['echo "foo"', 'echo "bar"'], [1, 2]),
        ("commands_with_mixed_comments_1.txt", ['echo "foo"', 'echo "bar"'], [1, 3]),
        ("commands_with_mixed_comments_2.txt", ['echo "foo"', 'echo "bar"'], [2, 4]),
        ("command_after_first_non_comment_line.txt", ['echo "foo"', 'echo "bar"'], [2, 4]),
        ("bang_not_at_start_of_line.txt", ['echo "foo"'], [2]),
        ("space_between_bang_and_command.txt", ['echo "foo"'], [1]),
        ("multiple_spaces_between_band_and_command.txt", ['echo "foo"'], [1]),
        ("app_commands_to_ignore.txt", [], []),
    ],
)
def test_extract_app_commands_from_file(filename, expected_commands, expected_line_numbers):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    test_file_path = os.path.join(dir_path, "testdata", "app_commands", filename)

    res = _extract_commands_from_file(file_name=test_file_path)

    assert res.file == test_file_path
    assert res.commands == expected_commands
    assert res.line_numbers == expected_line_numbers


def test_execute_app_commands_runs_single_command(capfd):
    cl = CommandLines(
        file="foo.txt",
        commands=['echo "foo"'],
        line_numbers=[1],
    )
    _execute_app_commands(cl)
    out, _ = capfd.readouterr()
    assert "foo" in out


def test_execute_app_commands_runs_multiple_commands(capfd):
    cl = CommandLines(
        file="foo.txt",
        commands=['echo "foo"', 'echo "bar"'],
        line_numbers=[1, 2],
    )
    _execute_app_commands(cl)
    out, _ = capfd.readouterr()
    assert "foo" in out
    assert "bar" in out


@pytest.mark.skipif(sys.platform.startswith("win"), reason="env command is not available on windows")
def test_execute_app_commands_runs_with_env_vars_patched(monkeypatch, capfd):
    monkeypatch.setenv("TEST_EXECUTE_APP_COMMANDS_RUNS_WITH_ENV_VARS_PATCHED", "TRUE")
    cl = CommandLines(
        file="foo.txt",
        commands=["env"],
        line_numbers=[1],
    )
    _execute_app_commands(cl)
    out, _ = capfd.readouterr()
    assert "TEST_EXECUTE_APP_COMMANDS_RUNS_WITH_ENV_VARS_PATCHED=TRUE" in out


def test_execute_app_commands_raises_appropriate_traceback_on_error(capfd):
    cl = CommandLines(
        file="foo.txt",
        commands=['echo "foo"', 'CommandDoesNotExist "somearg"'],
        line_numbers=[1, 3],
    )
    with pytest.raises(
        MisconfigurationException,
        match='There was a problem on line 3 of foo.txt while executing the command: CommandDoesNotExist "somearg"',
    ):
        _execute_app_commands(cl)
    out, err = capfd.readouterr()
    assert "foo" in out
    if sys.platform.startswith("linux"):
        assert "CommandDoesNotExist: not found" in err
    elif sys.platform.startswith("darwin"):
        assert "CommandDoesNotExist: command not found" in err
    else:
        assert "CommandDoesNotExist' is not recognized" in err
