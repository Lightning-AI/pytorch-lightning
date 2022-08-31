import os

from lightning_cloud.openapi.rest import ApiException

from lightning_app.testing.testing import _fetch_logs
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.network import LightningClient


def pytest_timeout_cancel_timer(item):
    """This hook fetches and prints the logs when timeout triggers."""

    if item.name.startswith("test_") and item.name.endswith("_cloud"):
        name = os.getenv("LIGHTNING_APP_NAME")
        print(f"Timeout was triggered. Fetching all the logs for the App {name}.")

        client = LightningClient()
        project = _get_project(client)

        lightning_apps = [
            app
            for app in client.lightningapp_instance_service_list_lightningapp_instances(
                project.project_id
            ).lightningapps
            if app.name == name
        ]

        if not lightning_apps:
            return True

        assert len(lightning_apps) == 1

        lightning_app = lightning_apps[0]

        print("##################################################")

        for log in _fetch_logs(component_names=None, client=client, app_id=lightning_app.id, project=project):
            print(log)

        print("##################################################")

        try:
            res = client.lightningapp_instance_service_delete_lightningapp_instance(
                project_id=project.project_id,
                id=lightning_app.id,
            )
            assert res == {}
        except ApiException as e:
            print(f"Failed to delete {name}. Exception {e}")

    return True
