import shutil
from unittest import mock
from unittest.mock import MagicMock

from click.testing import CliRunner
from lightning_cloud.openapi import (
    Externalv1LightningappInstance,
    Externalv1Lightningwork,
    V1ClusterStatus,
    V1GetClusterResponse,
    V1LightningappInstanceSpec,
)

from lightning_app.cli.lightning_cli import ssh


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("inquirer.prompt")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.list_apps")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.list_components")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.get_app")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.get_cluster")
@mock.patch("os.execv")
def test_ssh_no_arguments(
    os_execv: mock.MagicMock,
    get_cluster: mock.MagicMock,
    get_app: mock.MagicMock,
    list_components: mock.MagicMock,
    list_apps: mock.MagicMock,
    list_prompt: mock.MagicMock,
):
    app_instance = Externalv1LightningappInstance(
        id="test1234",
        name="test",
        spec=V1LightningappInstanceSpec(cluster_id="clusterA"),
    )
    list_apps.return_value = [app_instance]
    list_components.return_value = [Externalv1Lightningwork(id="work1234", name="root.server")]
    get_app.return_value = app_instance
    get_cluster.return_value = V1GetClusterResponse(status=V1ClusterStatus(ssh_gateway_endpoint="ssh.lightning.ai"))
    list_prompt.side_effect = [{"app_name": "test"}, {"component_name": "root.server"}]

    runner = CliRunner()
    runner.invoke(ssh, [])

    ssh_path = shutil.which("ssh")
    os_execv.assert_called_once_with(ssh_path, ["-tt", "lightningwork-work1234@ssh.lightning.ai"])


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("inquirer.prompt")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.list_apps")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.list_components")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.get_app")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.get_cluster")
@mock.patch("os.execv")
def test_ssh_app_preselected(
    os_execv: mock.MagicMock,
    get_cluster: mock.MagicMock,
    get_app: mock.MagicMock,
    list_components: mock.MagicMock,
    list_apps: mock.MagicMock,
    list_prompt: mock.MagicMock,
):
    app_instance = Externalv1LightningappInstance(
        id="test1234",
        name="test",
        spec=V1LightningappInstanceSpec(cluster_id="clusterA"),
    )
    list_apps.return_value = [app_instance]
    list_components.return_value = [Externalv1Lightningwork(id="work1234", name="root.server")]
    get_app.return_value = app_instance
    get_cluster.return_value = V1GetClusterResponse(status=V1ClusterStatus(ssh_gateway_endpoint="ssh.lightning.ai"))
    list_prompt.return_value = {"component_name": "root.server"}

    runner = CliRunner()
    runner.invoke(ssh, ["--app-name", "test"])

    ssh_path = shutil.which("ssh")
    os_execv.assert_called_once_with(ssh_path, ["-tt", "lightningwork-work1234@ssh.lightning.ai"])


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.cli.cmd_apps._AppManager.list_apps")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.list_components")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.get_app")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.get_cluster")
@mock.patch("os.execv")
def test_ssh_app_and_component_preselected(
    os_execv: mock.MagicMock,
    get_cluster: mock.MagicMock,
    get_app: mock.MagicMock,
    list_components: mock.MagicMock,
    list_apps: mock.MagicMock,
):
    app_instance = Externalv1LightningappInstance(
        id="test1234",
        name="test",
        spec=V1LightningappInstanceSpec(cluster_id="clusterA"),
    )
    list_apps.return_value = [app_instance]
    list_components.return_value = [Externalv1Lightningwork(id="work1234", name="root.server")]
    get_app.return_value = app_instance
    get_cluster.return_value = V1GetClusterResponse(status=V1ClusterStatus(ssh_gateway_endpoint="ssh.lightning.ai"))

    runner = CliRunner()
    runner.invoke(ssh, ["--app-name", "test", "--component-name", "root.server"])

    ssh_path = shutil.which("ssh")
    os_execv.assert_called_once_with(ssh_path, ["-tt", "lightningwork-work1234@ssh.lightning.ai"])


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.cli.cmd_apps._AppManager.list_apps")
@mock.patch("click.ClickException")
def test_ssh_unknown_app(
    click_exception: mock.MagicMock,
    list_apps: mock.MagicMock,
):
    app_instance = Externalv1LightningappInstance(
        id="test1234",
        name="test-different-name",
        spec=V1LightningappInstanceSpec(cluster_id="clusterA"),
    )
    list_apps.return_value = [app_instance]

    runner = CliRunner()
    runner.invoke(ssh, ["--app-name", "unknown-app-name"])

    click_exception.assert_called_once()


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.cli.cmd_apps._AppManager.list_apps")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.list_components")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.get_app")
@mock.patch("lightning_app.cli.cmd_apps._AppManager.get_cluster")
@mock.patch("click.ClickException")
def test_ssh_unknown_component(
    click_exception: mock.MagicMock,
    get_cluster: mock.MagicMock,
    get_app: mock.MagicMock,
    list_components: mock.MagicMock,
    list_apps: mock.MagicMock,
):
    app_instance = Externalv1LightningappInstance(
        id="test1234",
        name="test",
        spec=V1LightningappInstanceSpec(cluster_id="clusterA"),
    )
    list_apps.return_value = [app_instance]
    list_components.return_value = [Externalv1Lightningwork(id="work1234", name="root.server")]
    get_app.return_value = app_instance
    get_cluster.return_value = V1GetClusterResponse(status=V1ClusterStatus(ssh_gateway_endpoint="ssh.lightning.ai"))

    runner = CliRunner()
    runner.invoke(ssh, ["--app-name", "test", "--component-name", "rot.server"])

    click_exception.assert_called_once()
