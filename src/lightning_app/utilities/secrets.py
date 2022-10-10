from typing import Dict, Iterable

from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.network import LightningClient


def _names_to_ids(secret_names: Iterable[str]) -> Dict[str, str]:
    """Returns the name/ID pair for each given Secret name.

    Raises a `ValueError` if any of the given Secret names do not exist.
    """
    lightning_client = LightningClient()

    project = _get_project(lightning_client)
    secrets = lightning_client.secret_service_list_secrets(project_id=project.project_id)

    secret_names_to_ids: Dict[str, str] = {}
    for secret in secrets.secrets:
        if secret.name in secret_names:
            secret_names_to_ids[secret.name] = secret.id

    for secret_name in secret_names:
        if secret_name not in secret_names_to_ids.keys():
            raise ValueError(f"Secret with name '{secret_name}' not found")

    return secret_names_to_ids
