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

from typing import Dict, Iterable

from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.network import LightningClient


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
