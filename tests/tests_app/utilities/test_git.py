import sys

import pytest

from lightning.app.utilities.git import (
    check_github_repository,
    check_if_remote_head_is_different,
    execute_git_command,
    get_dir_name,
    get_git_relative_path,
    has_uncommitted_files,
)


@pytest.mark.skipif(sys.platform == "win32", reason="Don't run on windows")
def test_execute_git_command():
    res = execute_git_command(["pull"])
    assert res

    assert get_dir_name() == "lightning"

    assert check_github_repository()

    if sys.platform == "win32":
        assert get_git_relative_path(__file__) == "tests\\tests_app\\utilities\\test_git.py"
    else:
        assert get_git_relative_path(__file__) == "tests/tests_app/utilities/test_git.py"

    # this commands can be either True or False based on dev.
    check_if_remote_head_is_different()
    has_uncommitted_files()
