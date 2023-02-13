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
import re
import shutil
import subprocess
import sys
from typing import Dict, Optional, Tuple

import click
import requests
from packaging.version import Version

from lightning.app.core.constants import LIGHTNING_APPS_PUBLIC_REGISTRY, LIGHTNING_COMPONENT_PUBLIC_REGISTRY
from lightning.app.utilities.app_helpers import Logger

logger = Logger(__name__)


@click.group(name="install")
def install() -> None:
    """Install Lightning AI selfresources."""
    pass


@install.command("app")
@click.argument("name", type=str)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="disables prompt to ask permission to create env and run install cmds",
)
@click.option(
    "--version",
    "-v",
    type=str,
    help="Specify the version to install. By default it uses 'latest'",
    default="latest",
    show_default=True,
)
@click.option(
    "--overwrite",
    "-f",
    is_flag=True,
    default=False,
    help="When set, overwrite the app directory without asking if it already exists.",
)
def install_app(name: str, yes: bool, version: str, overwrite: bool = False) -> None:
    _install_app_command(name, yes, version, overwrite=overwrite)


@install.command("component")
@click.argument("name", type=str)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="disables prompt to ask permission to create env and run install cmds",
)
@click.option(
    "--version",
    "-v",
    type=str,
    help="Specify the version to install. By default it uses 'latest'",
    default="latest",
    show_default=True,
)
def install_component(name: str, yes: bool, version: str) -> None:
    _install_component_command(name, yes, version)


def _install_app_command(name: str, yes: bool, version: str, overwrite: bool = False) -> None:
    if "github.com" in name:
        if version != "latest":
            logger.warn(
                "When installing from GitHub, only the 'latest' version is supported. "
                f"The provided version ({version}) will be ignored."
            )
        return non_gallery_app(name, yes, overwrite=overwrite)
    else:
        return gallery_app(name, yes, version, overwrite=overwrite)


def _install_component_command(name: str, yes: bool, version: str, overwrite: bool = False) -> None:
    if "github.com" in name:
        if version != "latest":
            logger.warn(
                "When installing from GitHub, only the 'latest' version is supported. "
                f"The provided version ({version}) will be ignored."
            )
        return non_gallery_component(name, yes)
    else:
        return gallery_component(name, yes, version)


def gallery_apps_and_components(
    name: str, yes_arg: bool, version_arg: str, cwd: str = None, overwrite: bool = False
) -> Optional[str]:

    try:
        org, app_or_component = name.split("/")
    except Exception:
        return None

    entry, kind = _resolve_entry(name, version_arg)

    if kind == "app":
        # give the user the chance to do a manual install
        source_url, git_url, folder_name, git_sha = _show_install_app_prompt(
            entry, app_or_component, org, yes_arg, resource_type="app"
        )
        # run installation if requested
        _install_app_from_source(source_url, git_url, folder_name, cwd=cwd, overwrite=overwrite, git_sha=git_sha)

        return os.path.join(os.getcwd(), *entry["appEntrypointFile"].split("/"))

    elif kind == "component":
        # give the user the chance to do a manual install
        source_url, git_url, folder_name, git_sha = _show_install_app_prompt(
            entry, app_or_component, org, yes_arg, resource_type="component"
        )
        if "@" in git_url:
            git_url = git_url.split("git+")[1].split("@")[0]
        # run installation if requested
        _install_app_from_source(source_url, git_url, folder_name, cwd=cwd, overwrite=overwrite, git_sha=git_sha)

        return os.path.join(os.getcwd(), *entry["entrypointFile"].split("/"))

    return None


def gallery_component(name: str, yes_arg: bool, version_arg: str, cwd: str = None) -> str:
    # make sure org/component-name name is correct
    org, component = _validate_name(name, resource_type="component", example="lightning/LAI-slack-component")

    # resolve registry (orgs can have a private registry through their environment variables)
    registry_url = _resolve_component_registry()

    # load the component resource
    component_entry = _resolve_resource(registry_url, name=name, version_arg=version_arg, resource_type="component")

    # give the user the chance to do a manual install
    git_url = _show_install_component_prompt(component_entry, component, org, yes_arg)

    # run installation if requested
    _install_component_from_source(git_url)

    return os.path.join(os.getcwd(), component_entry["entrypointFile"])


def non_gallery_component(gh_url: str, yes_arg: bool, cwd: str = None) -> None:
    # give the user the chance to do a manual install
    git_url = _show_non_gallery_install_component_prompt(gh_url, yes_arg)

    # run installation if requested
    _install_component_from_source(git_url)


def gallery_app(name: str, yes_arg: bool, version_arg: str, cwd: str = None, overwrite: bool = False) -> str:
    # make sure org/app-name syntax is correct
    org, app = _validate_name(name, resource_type="app", example="lightning/quick-start")

    # resolve registry (orgs can have a private registry through their environment variables)
    registry_url = _resolve_app_registry()

    # load the app resource
    app_entry = _resolve_resource(registry_url, name=name, version_arg=version_arg, resource_type="app")

    # give the user the chance to do a manual install
    source_url, git_url, folder_name, git_sha = _show_install_app_prompt(
        app_entry, app, org, yes_arg, resource_type="app"
    )

    # run installation if requested
    _install_app_from_source(source_url, git_url, folder_name, cwd=cwd, overwrite=overwrite, git_sha=git_sha)

    return os.path.join(os.getcwd(), folder_name, app_entry["appEntrypointFile"])


def non_gallery_app(gh_url: str, yes_arg: bool, cwd: str = None, overwrite: bool = False) -> None:
    # give the user the chance to do a manual install
    repo_url, folder_name = _show_non_gallery_install_app_prompt(gh_url, yes_arg)

    # run installation if requested
    _install_app_from_source(repo_url, repo_url, folder_name, cwd=cwd, overwrite=overwrite)


def _show_install_component_prompt(entry: Dict[str, str], component: str, org: str, yes_arg: bool) -> str:
    git_url = entry["gitUrl"]

    # yes arg does not prompt the user for permission to install anything
    # automatically creates env and sets up the project
    if yes_arg:
        return git_url

    prompt = f"""
    ⚡ Installing Lightning component ⚡

    component name : {component}
    developer      : {org}

    Installation runs the following command for you:

    pip install {git_url}
    """
    logger.info(prompt)

    try:
        value = input("\nPress enter to continue:   ")
        value = value.strip().lower()
        should_install = len(value) == 0 or value in {"y", "yes", 1}
        if not should_install:
            raise KeyboardInterrupt()

        return git_url
    except KeyboardInterrupt:
        repo = entry["sourceUrl"]
        m = f"""
        ⚡ Installation aborted! ⚡

        Install the component yourself by visiting:
        {repo}
        """
        raise SystemExit(m)


def _show_non_gallery_install_component_prompt(gh_url: str, yes_arg: bool) -> str:
    if ".git@" not in gh_url:
        m = """
        Error, your github url must be in the following format:
        git+https://github.com/OrgName/repo-name.git@ALongCommitSHAString

        Example:
        git+https://github.com/Lightning-AI/LAI-slack-messenger.git@14f333456ffb6758bd19458e6fa0bf12cf5575e1
        """
        raise SystemExit(m)

    developer = gh_url.split("/")[3]
    component_name = gh_url.split("/")[4].split(".git")[0]
    repo_url = re.search(r"git\+(.*).git", gh_url).group(1)  # type: ignore

    # yes arg does not prompt the user for permission to install anything
    # automatically creates env and sets up the project
    if yes_arg:
        return gh_url

    prompt = f"""
    ⚡ Installing Lightning component ⚡

    component name : {component_name}
    developer      : {developer}

    ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡
    WARNING: this is NOT an official Lightning Gallery component
    Install at your own risk
    ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡

    Installation runs the following command for you:

    pip install {gh_url}
    """
    logger.info(prompt)

    try:
        value = input("\nPress enter to continue:   ")
        value = value.strip().lower()
        should_install = len(value) == 0 or value in {"y", "yes", 1}
        if not should_install:
            raise KeyboardInterrupt()

        return gh_url
    except KeyboardInterrupt:
        m = f"""
        ⚡ Installation aborted! ⚡

        Install the component yourself by visiting:
        {repo_url}
        """
        raise SystemExit(m)


def _show_install_app_prompt(
    entry: Dict[str, str], app: str, org: str, yes_arg: bool, resource_type: str
) -> Tuple[str, str, str, Optional[str]]:
    source_url = entry["sourceUrl"]  # This URL is used only to display the repo and extract folder name
    full_git_url = entry["gitUrl"]  # Used to clone the repo (can include tokens for private repos)
    git_url_parts = full_git_url.split("#ref=")
    git_url = git_url_parts[0]
    git_sha = git_url_parts[1] if len(git_url_parts) == 2 else None

    folder_name = source_url.split("/")[-1]

    # yes arg does not prompt the user for permission to install anything
    # automatically creates env and sets up the project
    if yes_arg:
        return source_url, git_url, folder_name, git_sha

    prompt = f"""
    ⚡ Installing Lightning {resource_type} ⚡

    {resource_type} name : {app}
    developer: {org}

    Installation creates and runs the following commands for you:

    git clone {source_url}
    cd {folder_name}
    pip install -r requirements.txt
    pip install -e .
    """
    logger.info(prompt)

    try:
        value = input("\nPress enter to continue:   ")
        value = value.strip().lower()
        should_install = len(value) == 0 or value in {"y", "yes", 1}
        if not should_install:
            raise KeyboardInterrupt()

        return source_url, git_url, folder_name, git_sha
    except KeyboardInterrupt:
        repo = entry["sourceUrl"]
        m = f"""
        ⚡ Installation aborted! ⚡

        Install the {resource_type} yourself by visiting:
        {repo}
        """
        raise SystemExit(m)


def _show_non_gallery_install_app_prompt(gh_url: str, yes_arg: bool) -> Tuple[str, str]:
    try:
        if gh_url.endswith(".git"):
            # folder_name when it's a GH url with .git
            folder_name = gh_url.split("/")[-1]
            folder_name = folder_name[:-4]
        else:
            # the last part of the url is the folder name otherwise
            folder_name = gh_url.split("/")[-1]

        org = re.search(r"github.com\/(.*)\/", gh_url).group(1)  # type: ignore
    except Exception as e:  # noqa
        m = """
        Your github url is not supported. Here's the supported format:
        https://github.com/YourOrgName/your-repo-name

        Example:
        https://github.com/Lightning-AI/lightning
        """
        raise SystemExit("")

    # yes arg does not prompt the user for permission to install anything
    # automatically creates env and sets up the project
    if yes_arg:
        return gh_url, folder_name

    prompt = f"""
    ⚡ Installing Lightning app ⚡

    app source : {gh_url}
    developer  : {org}

    ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡
    WARNING: this is NOT an official Lightning Gallery app
    Install at your own risk
    ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡

    Installation creates and runs the following commands for you:

    git clone {gh_url}
    cd {folder_name}
    pip install -r requirements.txt
    pip install -e .
    """
    logger.info(prompt)

    try:
        value = input("\nPress enter to continue:   ")
        value = value.strip().lower()
        should_install = len(value) == 0 or value in {"y", "yes", 1}
        if not should_install:
            raise KeyboardInterrupt()

        return gh_url, folder_name
    except KeyboardInterrupt:
        m = f"""
        ⚡ Installation aborted! ⚡

        Install the app yourself by visiting {gh_url}
        """
        raise SystemExit(m)


def _validate_name(name: str, resource_type: str, example: str) -> Tuple[str, str]:
    # ensure resource identifier is properly formatted
    try:
        org, resource = name.split("/")
    except Exception as e:  # noqa
        m = f"""
        {resource_type} name format must have organization/{resource_type}-name

        Examples:
        {example}
        user/{resource_type}-name

        You passed in: {name}
        """
        raise SystemExit(m)
    m = f"""
    ⚡ Installing Lightning {resource_type} ⚡
    {resource_type} name: {resource}
    developer: {org}
    """
    return org, resource


def _resolve_entry(name, version_arg) -> Tuple[Optional[Dict], Optional[str]]:
    entry = None
    kind = None

    # resolve registry (orgs can have a private registry through their environment variables)
    registry_url = _resolve_app_registry()

    # load the app resource
    entry = _resolve_resource(registry_url, name=name, version_arg=version_arg, resource_type="app", raise_error=False)

    if not entry:

        registry_url = _resolve_component_registry()

        # load the component resource
        entry = _resolve_resource(
            registry_url, name=name, version_arg=version_arg, resource_type="component", raise_error=False
        )
        kind = "component" if entry else None

    else:
        kind = "app"

    return entry, kind


def _resolve_resource(
    registry_url: str, name: str, version_arg: str, resource_type: str, raise_error: bool = True
) -> Dict[str, str]:
    gallery_entries = []
    try:
        response = requests.get(registry_url)
        data = response.json()

        if resource_type == "app":
            gallery_entries = [a for a in data["apps"] if a["canDownloadSourceCode"]]

        elif resource_type == "component":
            gallery_entries = data["components"]
    except requests.ConnectionError:
        m = f"""
        Network connection error, could not load list of available Lightning {resource_type}s.

        Try again when you have a network connection!
        """
        sys.tracebacklimit = 0
        raise SystemError(m)

    entries = []
    all_versions = []
    for x in gallery_entries:
        if name == x["name"]:
            entries.append(x)
            all_versions.append(x["version"])

    if len(entries) == 0:
        if raise_error:
            raise SystemExit(f"{resource_type}: '{name}' is not available on ⚡ Lightning AI ⚡")
        else:
            return None

    entry = None
    if version_arg == "latest":
        entry = max(entries, key=lambda app: Version(app["version"]))
    else:
        for e in entries:
            if e["version"] == version_arg:
                entry = e
                break
    if entry is None and raise_error:
        if raise_error:
            raise Exception(
                f"{resource_type}: 'Version {version_arg} for {name}' is not available on ⚡ Lightning AI ⚡. "
                f"Here is the list of all availables versions:{os.linesep}{os.linesep.join(all_versions)}"
            )
        else:
            return None

    return entry


def _install_with_env(repo_url: str, folder_name: str, cwd: str = None) -> None:
    if not cwd:
        cwd = os.getcwd()

    # clone repo
    logger.info(f"⚡ RUN: git clone {repo_url}")
    subprocess.call(["git", "clone", repo_url])

    # step into the repo folder
    os.chdir(f"{folder_name}")
    cwd = os.getcwd()

    # create env
    logger.info(f"⚡ CREATE: virtual env at {cwd}")
    subprocess.call(["python", "-m", "venv", cwd])

    # activate and install reqs
    # TODO: remove shell=True... but need to run command in venv
    logger.info("⚡ RUN: install requirements (pip install -r requirements.txt)")
    subprocess.call("source bin/activate && pip install -r requirements.txt", shell=True)

    # install project
    # TODO: remove shell=True... but need to run command in venv
    logger.info("⚡ RUN: setting up project (pip install -e .)")
    subprocess.call("source bin/activate && pip install -e .", shell=True)

    m = f"""
    ⚡ Installed! ⚡ to use your app
        go into the folder: cd {folder_name}
    activate the environment: source bin/activate
                run the app: lightning run app [the_app_file.py]
    """
    logger.info(m)


def _install_app_from_source(
    source_url: str, git_url: str, folder_name: str, cwd: str = None, overwrite: bool = False, git_sha: str = None
) -> None:
    """Installing lighting app from the `git_url`

    Args:
        source_url:
            source repo url without any tokens and params, this param is used only for displaying
        git_url:
            repo url that is used to clone, this can contain tokens
        folder_name:
            where to clone the repo ?
        cwd:
            Working director. If not specified, current working directory is used.
        overwrite:
            If true, overwrite the app directory without asking if it already exists
        git_sha:
            The git_sha for checking out the git repo of the app.
    """

    if not cwd:
        cwd = os.getcwd()

    destination = os.path.join(cwd, folder_name)
    if os.path.exists(destination):
        if not overwrite:
            raise SystemExit(
                f"Folder {folder_name} exists, please delete it and try again, "
                f"or force to overwrite the existing folder by passing `--overwrite`.",
            )
        shutil.rmtree(destination)
    # clone repo
    logger.info(f"⚡ RUN: git clone {source_url}")
    try:
        subprocess.check_output(["git", "clone", git_url], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        if "Repository not found" in str(e.output):
            m = f"""
            Looks like the github url was not found or doesn't exist. Do you have a typo?
            {source_url}
            """
            raise SystemExit(m)
        else:
            raise Exception(e)

    # step into the repo folder
    os.chdir(f"{folder_name}")
    cwd = os.getcwd()

    try:
        if git_sha:
            subprocess.check_output(["git", "checkout", git_sha], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        if "did not match any" in str(e.output):
            raise SystemExit("Looks like the git SHA is not valid or doesn't exist in app repo.")
        else:
            raise Exception(e)

    # activate and install reqs
    # TODO: remove shell=True... but need to run command in venv
    logger.info("⚡ RUN: install requirements (pip install -r requirements.txt)")
    subprocess.call("pip install -r requirements.txt", shell=True)

    # install project
    # TODO: remove shell=True... but need to run command in venv
    logger.info("⚡ RUN: setting up project (pip install -e .)")
    subprocess.call("pip install -e .", shell=True)

    m = f"""
    ⚡ Installed! ⚡ to use your app:

    cd {folder_name}
    lightning run app app.py
    """
    logger.info(m)


def _install_component_from_source(git_url: str) -> None:
    logger.info("⚡ RUN: pip install")

    out = subprocess.check_output(["pip", "install", git_url])
    possible_success_message = [x for x in str(out).split("\\n") if "Successfully installed" in x]
    if len(possible_success_message) > 0:
        uninstall_step = possible_success_message[0]
        uninstall_step = re.sub("Successfully installed", "", uninstall_step).strip()
        uninstall_step = re.sub("-0.0.0", "", uninstall_step).strip()
        m = """
        ⚡ Installed! ⚡

        to use your component:
        from the_component import TheClass

        make sure to add this entry to your Lightning APP requirements.txt file:
        {git_url}

        if you want to uninstall, run this command:
        pip uninstall {uninstall_step}
        """
        logger.info(m)


def _resolve_app_registry() -> str:
    registry = os.environ.get("LIGHTNING_APP_REGISTRY", LIGHTNING_APPS_PUBLIC_REGISTRY)
    return registry


def _resolve_component_registry() -> str:
    registry = os.environ.get("LIGHTNING_COMPONENT_REGISTRY", LIGHTNING_COMPONENT_PUBLIC_REGISTRY)
    return registry
