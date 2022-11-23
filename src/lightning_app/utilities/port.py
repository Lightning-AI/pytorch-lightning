import os
from typing import Optional

from lightning_cloud.openapi import AppinstancesIdBody, Externalv1LightningappInstance

from lightning_app.utilities.network import LightningClient


def _find_lit_app_port(default_port: int) -> int:
    """Make a request to the cloud controlplane to find a disabled port of the flow, enable it and return it."""

    app_id = os.getenv("LIGHTNING_CLOUD_APP_ID", None)
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
    enable_multiple_works_in_default_container = bool(int(os.getenv("ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER", "0")))

    if not app_id or not project_id or not enable_multiple_works_in_default_container:
        return default_port

    assert project_id

    client = LightningClient()
    list_apps_resp = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id)
    lit_app: Optional[Externalv1LightningappInstance] = None

    for lightningapp in list_apps_resp.lightningapps:
        if lightningapp.id == app_id:
            lit_app = lightningapp

    assert lit_app

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

    assert found_nc

    # Note: This is required for the framework to know we need to use the CloudMultiProcessRuntime.
    os.environ["APP_SERVER_HOST"] = f"https:/{found_nc.host}"

    return found_nc.port


def open_port():
    """Make a request to the cloud controlplane to open a port of the flow."""
    app_id = os.getenv("LIGHTNING_CLOUD_APP_ID", None)
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)

    if not app_id or not project_id:
        raise Exception("The app_id and project_id should be defined.")

    assert project_id

    client = LightningClient()
    list_apps_resp = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id)
    lit_app: Optional[Externalv1LightningappInstance] = None

    for lightningapp in list_apps_resp.lightningapps:
        if lightningapp.id == app_id:
            lit_app = lightningapp

    assert lit_app

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

    assert found_nc

    return found_nc


def close_port(port: int) -> None:
    """Make a request to the cloud controlplane to close a port of the flow."""

    app_id = os.getenv("LIGHTNING_CLOUD_APP_ID", None)
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)

    if not app_id or not project_id:
        raise Exception("The app_id and project_id should be defined.")

    assert project_id

    client = LightningClient()
    list_apps_resp = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id)
    lit_app: Optional[Externalv1LightningappInstance] = None

    for lightningapp in list_apps_resp.lightningapps:
        if lightningapp.id == app_id:
            lit_app = lightningapp

    assert lit_app

    found_nc = None

    for nc in lit_app.spec.network_config:
        if nc.port == port:
            nc.enable = False
            found_nc = nc
            break

    client.lightningapp_instance_service_update_lightningapp_instance(
        project_id=project_id,
        id=lit_app.id,
        body=AppinstancesIdBody(name=lit_app.name, spec=lit_app.spec),
    )

    assert found_nc
