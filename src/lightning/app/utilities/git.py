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

import subprocess
from pathlib import Path
from typing import List, Union

# TODO - github utilities are already defined in GridSDK use that?


def execute_git_command(args: List[str], cwd=None) -> str:
    """Executes a git command. This is expected to return a single string back.

    Returns
    -------
    output: str
        String combining stdout and stderr.
    """
    process = subprocess.run(["git"] + args, capture_output=True, text=True, cwd=cwd, check=False)
    output = process.stdout.strip() + process.stderr.strip()
    return output


def get_dir_name(cwd=None) -> str:
    github_repository = execute_git_command(["config", "--get", "remote.origin.url"], cwd=cwd)
    if github_repository and "github.com" in github_repository:
        return github_repository.split("/")[-1].split(".")[0]
    raise RuntimeError("Only work with github repositories.")


def check_github_repository(cwd=None) -> bool:
    """Checks if the active directory is a GitHub repository."""
    github_repository = execute_git_command(["config", "--get", "remote.origin.url"], cwd=cwd)

    if not github_repository or "github.com" not in github_repository:
        return False
    return True


def get_git_relative_path(file: Union[str, Path]) -> str:
    if not check_github_repository():
        raise ValueError("Not a GitHub repository.")
    """  Finds the relative path of the file to the git root. """
    abs_path = Path(file).absolute()
    repository_path = execute_git_command(["rev-parse", "--show-toplevel"])
    return str(abs_path.relative_to(repository_path))


def check_if_remote_head_is_different() -> Union[bool, None]:
    """Checks if remote git repository is different than the version available locally.

    This only compares the local SHA to the HEAD commit of a given branch. This check won't be used if user isn't in a
    HEAD locally.
    """
    # Check SHA values.
    local_sha = execute_git_command(["rev-parse", "@"])
    remote_sha = execute_git_command(["rev-parse", r"@{u}"])
    base_sha = execute_git_command(["merge-base", "@", r"@{u}"])

    # Whenever a SHA is not avaialble, just return.
    if any("fatal" in f for f in (local_sha, remote_sha, base_sha)):
        return None

    is_different = True
    if local_sha in (remote_sha, base_sha):
        is_different = False

    return is_different


def has_uncommitted_files() -> bool:
    """Checks if user has uncommited files in local repository.

    If there are uncommited files, then show a prompt indicating that uncommited files exist locally.
    """
    files = execute_git_command(["update-index", "--refresh"])
    return bool(files)
