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

import json
import random
import string
from typing import List, Optional

from lightning_cloud.openapi import V1CreateSSHPublicKeyRequest, V1SSHPublicKey
from rich.console import Console
from rich.table import Table

from lightning_app.cli.core import Formatable
from lightning_app.utilities.network import LightningClient


class _SSHKeyList(Formatable):
    def __init__(self, ssh_keys: List[V1SSHPublicKey]):
        self.ssh_keys = ssh_keys

    def as_json(self) -> str:
        return json.dumps(self.ssh_keys)

    def as_table(self) -> Table:
        table = Table("id", "public_key", "created", show_header=True, header_style="bold green")
        for ssh_key in self.ssh_keys:
            table.add_row(
                ssh_key.id,
                ssh_key.public_key,
                ssh_key.created_at.strftime("%Y-%m-%d"),
            )
        return table


class _SSHKeyManager:
    """_SSHKeyManager implements API calls specific to Lightning AI SSH-Keys."""

    def __init__(self) -> None:
        self.api_client = LightningClient(retry=False)

    def get_ssh_keys(self) -> _SSHKeyList:
        resp = self.api_client.s_sh_public_key_service_list_ssh_public_keys()
        return _SSHKeyList(resp.ssh_keys)

    def list(self) -> None:
        ssh_keys = self.get_ssh_keys()
        console = Console()
        console.print(ssh_keys.as_table())

    def add_key(self, public_key: str, name: Optional[str], comment: Optional[str]) -> None:
        key_name = name if name is not None else "-".join(random.choice(string.ascii_lowercase) for _ in range(5))
        self.api_client.s_sh_public_key_service_create_ssh_public_key(
            V1CreateSSHPublicKeyRequest(
                name=key_name,
                public_key=public_key,
                comment=comment if comment is not None else key_name,
            )
        )

    def remove_key(self, key_id: str) -> None:
        self.api_client.s_sh_public_key_service_delete_ssh_public_key(key_id)
