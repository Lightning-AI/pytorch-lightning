import os
import sys
from unittest.mock import patch

from lightning.app.utilities.git import (
    check_github_repository,
    check_if_remote_head_is_different,
    get_dir_name,
    get_git_relative_path,
    has_uncommitted_files,
)


def mock_execute_git_command(args, cwd=None) -> str:
    if args == ["config", "--get", "remote.origin.url"]:
        return "https://github.com/Lightning-AI/lightning.git"

    if args == ["rev-parse", "--show-toplevel"]:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    if args == ["update-index", "--refresh"]:
        return ""

    if args == ["rev-parse", "@"]:
        return "local-sha"

    if args == ["rev-parse", r"@{u}"] or args == ["merge-base", "@", r"@{u}"]:
        return "remote-sha"

    return "Error: Unexpected call"


@patch("lightning.app.utilities.git.execute_git_command", mock_execute_git_command)
def test_execute_git_command():
    assert get_dir_name() == "lightning"

    assert check_github_repository()

    if sys.platform == "win32":
        assert get_git_relative_path(__file__) == "tests\\tests_app\\utilities\\test_git.py"
    else:
        assert get_git_relative_path(__file__) == "tests/tests_app/utilities/test_git.py"

    assert check_if_remote_head_is_different()

    assert not has_uncommitted_files()
