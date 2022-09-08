from typing import Dict, List

from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.network import LightningClient


def _names_to_ids(secret_names: List[str]) -> Dict[str, str]:
    """Returns the name/ID pair for each given Secret name."""
    lightning_client = LightningClient()

    project = _get_project(lightning_client)
    secrets = lightning_client.secret_service_list_secrets(project.project_id)

    secret_names_to_ids: Dict[str, str] = {}
    for secret in secrets.secrets:
        if secret.name in secret_names:
            secret_names_to_ids[secret.name] = secret.id

    return secret_names_to_ids
