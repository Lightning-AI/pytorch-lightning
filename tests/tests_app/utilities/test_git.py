import sys

from lightning_app.utilities.git import (
    check_github_repository,
    check_if_remote_head_is_different,
    execute_git_command,
    get_dir_name,
    get_git_relative_path,
    has_uncommitted_files,
)


def test_execute_git_command():

    res = execute_git_command(["pull"])
    assert res

    assert get_dir_name() == "lightning-app"

    assert check_github_repository()

    if sys.platform == "win32":
        assert get_git_relative_path(__file__) == "tests\\utilities\\test_git.py"
    else:
        assert get_git_relative_path(__file__) == "tests/utilities/test_git.py"

    # this commands can be either True or False based on dev.
    check_if_remote_head_is_different()
    has_uncommitted_files()
