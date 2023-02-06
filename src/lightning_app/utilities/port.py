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
import socket
from typing import Optional

from lightning_cloud.openapi import AppinstancesIdBody, Externalv1LightningappInstance, V1NetworkConfig

from lightning_app.utilities.network import find_free_network_port, LightningClient


def is_port_in_use(port: int) -> bool:
    """Checks if the given port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _find_lit_app_port(default_port: int) -> int:
    """Make a request to the cloud controlplane to find a disabled port of the flow, enable it and return it."""
    app_id = os.getenv("LIGHTNING_CLOUD_APP_ID", None)
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
    enable_multiple_works_in_default_container = bool(int(os.getenv("ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER", "0")))

    if not app_id or not project_id or not enable_multiple_works_in_default_container:
        app_port = default_port

        # If the default port is not available, picks any other available one
        if is_port_in_use(default_port):
            app_port = find_free_network_port()

        return app_port

    client = LightningClient()
    list_apps_resp = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id)
    lit_app: Optional[Externalv1LightningappInstance] = None

    for lightningapp in list_apps_resp.lightningapps:
        if lightningapp.id == app_id:
            lit_app = lightningapp

    if not lit_app:
        raise RuntimeError(
            "App was not found. Please open an issue at https://github.com/lightning-AI/lightning/issues."
        )

    found_nc = None

    for nc in lit_app.spec.network_config:
        if not nc.enable:
            found_nc = nc
            nc.enable = True
            break

    client.lightningapp_instance_service_update_lightningapp_instance(
        project_id=project_id,
        id=lit_app.id,
        body=AppinstancesIdBody(name=lit_app.name, spec=lit_app.spec),
    )

    if not found_nc:
        raise RuntimeError(
            "No available port was found. Please open an issue at https://github.com/lightning-AI/lightning/issues."
        )

    # Note: This is required for the framework to know we need to use the CloudMultiProcessRuntime.
    os.environ["APP_SERVER_HOST"] = f"https://{found_nc.host}"

    return found_nc.port


def enable_port() -> V1NetworkConfig:
    """Make a request to the cloud controlplane to open a port of the flow."""
    app_id = os.getenv("LIGHTNING_CLOUD_APP_ID", None)
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)

    if not app_id or not project_id:
        raise Exception("The app_id and project_id should be defined.")

    client = LightningClient()
    list_apps_resp = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id)
    lit_app: Optional[Externalv1LightningappInstance] = None

    for lightningapp in list_apps_resp.lightningapps:
        if lightningapp.id == app_id:
            lit_app = lightningapp

    if not lit_app:
        raise RuntimeError(
            "App was not found. Please open an issue at https://github.com/lightning-AI/lightning/issues."
        )

    found_nc = None

    for nc in lit_app.spec.network_config:
        if not nc.enable:
            found_nc = nc
            nc.enable = True
            break

    client.lightningapp_instance_service_update_lightningapp_instance(
        project_id=project_id,
        id=lit_app.id,
        body=AppinstancesIdBody(name=lit_app.name, spec=lit_app.spec),
    )

    if not found_nc:
        raise RuntimeError(
            "No available port was found. Please open an issue at https://github.com/lightning-AI/lightning/issues."
        )

    return found_nc


def disable_port(port: int, ignore_disabled: bool = True) -> None:
    """Make a request to the cloud controlplane to close a port of the flow."""

    app_id = os.getenv("LIGHTNING_CLOUD_APP_ID", None)
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)

    if not app_id or not project_id:
        raise Exception("The app_id and project_id should be defined.")

    client = LightningClient()
    list_apps_resp = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id)
    lit_app: Optional[Externalv1LightningappInstance] = None

    for lightningapp in list_apps_resp.lightningapps:
        if lightningapp.id == app_id:
            lit_app = lightningapp

    if not lit_app:
        raise RuntimeError(
            "App was not found. Please open an issue at https://github.com/lightning-AI/lightning/issues."
        )

    found_nc = None

    for nc in lit_app.spec.network_config:
        if nc.port == port:
            if not nc.enable and not ignore_disabled:
                raise RuntimeError(f"The port {port} was already disabled.")

            nc.enable = False
            found_nc = nc
            break

    client.lightningapp_instance_service_update_lightningapp_instance(
        project_id=project_id,
        id=lit_app.id,
        body=AppinstancesIdBody(name=lit_app.name, spec=lit_app.spec),
    )

    if not found_nc:
        ports = [nc.port for nc in lit_app.spec.network_config]
        raise ValueError(f"The provided port doesn't exists. Available ports are {ports}.")

    assert found_nc
