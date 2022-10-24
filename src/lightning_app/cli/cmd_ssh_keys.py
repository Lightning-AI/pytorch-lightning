import json
from typing import List

from lightning_cloud.openapi import V1CreateSSHPublicKeyRequest, V1SSHPublicKey
from rich.console import Console
from rich.table import Table

from lightning_app.cli.core import Formatable
from lightning_app.utilities.network import LightningClient


class SSHKeyList(Formatable):
    def __init__(self, ssh_keys: List[V1SSHPublicKey]):
        self.ssh_keys = ssh_keys

    def as_json(self) -> str:
        return json.dumps(self.ssh_keys)

    def as_table(self) -> Table:
        table = Table("id", "public_key", "created", show_header=True)
        for ssh_key in self.ssh_keys:
            table.add_row(
                ssh_key.id,
                ssh_key.public_key,
                ssh_key.created_at.strftime("%Y-%m-%d"),
            )
        return table


class SSHKeyManager:
    """SSHKeyManager implements API calls specific to Lightning AI SSH-Keys."""

    def __init__(self) -> None:
        self.api_client = LightningClient()

    def get_ssh_keys(self) -> SSHKeyList:
        resp = self.api_client.s_sh_public_key_service_list_ssh_public_keys()
        return SSHKeyList(resp.ssh_keys)

    def list(self) -> None:
        ssh_keys = self.get_ssh_keys()
        console = Console()
        console.print(ssh_keys.as_table())

    def add_key(self, name: str, public_key: str, comment: str) -> None:
        self.api_client.s_sh_public_key_service_create_ssh_public_key(
            V1CreateSSHPublicKeyRequest(
                name=name,
                public_key=public_key,
                comment=comment,
            )
        )

    def remove_key(self, key_id: str) -> None:
        self.api_client.s_sh_public_key_service_delete_ssh_public_key(key_id)
