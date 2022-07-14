import os

from lightning_cloud.openapi.rest import ApiException

from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.network import LightningClient

client = LightningClient()

try:
    PR_NUMBER = int(os.getenv("PR_NUMBER", None))
except (TypeError, ValueError):
    # Failed when the PR is running master or 'PR_NUMBER' isn't defined.
    PR_NUMBER = ""

APP_NAME = os.getenv("TEST_APP_NAME", "")

project = _get_project(client)
list_lightningapps = client.lightningapp_instance_service_list_lightningapp_instances(project.project_id)

print([lightningapp.name for lightningapp in list_lightningapps.lightningapps])

for lightningapp in list_lightningapps.lightningapps:
    if PR_NUMBER and APP_NAME and not lightningapp.name.startswith(f"test-{PR_NUMBER}-{APP_NAME}-"):
        continue
    print(f"Deleting {lightningapp.name}")
    try:
        res = client.lightningapp_instance_service_delete_lightningapp_instance(
            project_id=project.project_id,
            id=lightningapp.id,
        )
        assert res == {}
    except ApiException as e:
        print(f"Failed to delete {lightningapp.name}. Exception {e}")
