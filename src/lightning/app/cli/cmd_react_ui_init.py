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

from lightning.app.utilities.app_helpers import Logger

logger = Logger(__name__)


def react_ui(dest_dir: str = None) -> None:
    # verify all the prereqs for install are met
    _check_react_prerequisites()

    # copy template files to the dir
    _copy_and_setup_react_ui(dest_dir)


def _copy_and_setup_react_ui(dest_dir: str = None) -> None:
    logger.info("⚡ setting up react-ui template")
    path = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(path, "react-ui-template")

    if dest_dir is None:
        dest_dir = os.path.join(os.getcwd(), "react-ui")

    shutil.copytree(template_dir, dest_dir)

    logger.info("⚡ install react project deps")
    ui_path = os.path.join(dest_dir, "ui")
    subprocess.run(f"cd {ui_path} && yarn install", shell=True)

    logger.info("⚡ building react project")
    subprocess.run(f"cd {ui_path} && yarn build", shell=True)

    m = f"""
    ⚡⚡ react-ui created! ⚡⚡

    ⚡ Connect it to your component using `configure_layout`:

    # Use a LightningFlow or LightningWork
    class YourComponent(la.LightningFlow):
        def configure_layout(self):
            return la.frontend.StaticWebFrontend(Path(__file__).parent / "react-ui/src/dist")

    ⚡ run the example_app.py to see it live!
    lightning run app {dest_dir}/example_app.py

    """
    logger.info(m)


def _check_react_prerequisites() -> None:
    """Args are for test purposes only."""
    missing_msgs = []
    version_regex = r"\d{1,2}\.\d{1,2}\.\d{1,3}"

    logger.info("Checking pre-requisites for react")

    # make sure npm is installed
    npm_version = subprocess.check_output(["npm", "--version"])
    has_npm = bool(re.search(version_regex, str(npm_version)))
    npm_version = re.search(version_regex, str(npm_version))
    npm_version = None if npm_version is None else npm_version.group(0)

    if not has_npm:
        m = """
        This machine is missing 'npm'. Please install npm and rerun 'lightning init react-ui' again.

        Install instructions: https://docs.npmjs.com/downloading-and-installing-node-js-and-npm
        """
        missing_msgs.append(m)

    # make sure node is installed
    node_version = subprocess.check_output(["node", "--version"])
    has_node = bool(re.search(version_regex, str(node_version)))
    node_version = re.search(version_regex, str(node_version))
    node_version = None if node_version is None else node_version.group(0)

    if not has_node:
        m = """
        This machine is missing 'node'. Please install node and rerun 'lightning init react-ui' again.

        Install instructions: https://docs.npmjs.com/downloading-and-installing-node-js-and-npm
        """
        missing_msgs.append(m)

    # make sure yarn is installed
    yarn_version = subprocess.check_output(["yarn", "--version"])
    has_yarn = bool(re.search(version_regex, str(yarn_version)))
    yarn_version = re.search(version_regex, str(yarn_version))
    yarn_version = None if yarn_version is None else yarn_version.group(0)

    if not has_yarn:
        m = """
        This machine is missing 'yarn'. Please install npm+node first, then run

        npm install --global yarn

        Full install instructions: https://classic.yarnpkg.com/lang/en/docs/install/#mac-stable
        """
        missing_msgs.append(m)

    # exit or show success message
    if len(missing_msgs) > 0:
        missing_msg = "\n".join(missing_msgs)
        raise SystemExit(missing_msg)
    else:
        m = f"""
        found npm  version: {npm_version}
        found node version: {node_version}
        found yarn version: {yarn_version}

        Pre-requisites met!
        """
        logger.info(m)
