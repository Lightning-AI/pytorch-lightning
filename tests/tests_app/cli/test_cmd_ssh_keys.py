from unittest import mock
from unittest.mock import MagicMock

from lightning_cloud.openapi import V1CreateSSHPublicKeyRequest

from lightning_app.cli.cmd_ssh_keys import _SSHKeyManager


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.utilities.network.LightningClient.s_sh_public_key_service_create_ssh_public_key")
def test_add_ssh_key(api: mock.MagicMock):
    ssh_key_manager = _SSHKeyManager()
    ssh_key_manager.add_key(
        public_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAn8mYRnRG1banQcfXPCUC6R8FvQS+YgfIsl70/dD3Te your_email@example.com",  # noqa E501
        comment="test",
        name="test",
    )

    api.assert_called_once_with(
        V1CreateSSHPublicKeyRequest(
            public_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAn8mYRnRG1banQcfXPCUC6R8FvQS+YgfIsl70/dD3Te your_email@example.com",  # noqa E501
            comment="test",
            name="test",
        )
    )


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.utilities.network.LightningClient.s_sh_public_key_service_list_ssh_public_keys")
def test_list_ssh_keys(api: mock.MagicMock):
    ssh_key_manager = _SSHKeyManager()
    ssh_key_manager.list()

    api.assert_called_once()


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.utilities.network.LightningClient.s_sh_public_key_service_delete_ssh_public_key")
def test_delete_ssh_key(api: mock.MagicMock):
    ssh_key_manager = _SSHKeyManager()
    ssh_key_manager.remove_key(key_id="45AB3098-7ABB-46CA-AA04-5D390F8D6A81")

    api.assert_called_once_with("45AB3098-7ABB-46CA-AA04-5D390F8D6A81")
