from unittest import mock

from click.testing import CliRunner

from lightning_app.cli.lightning_cli import logs


@mock.patch("lightning_app.cli.lightning_cli.LightningClient")
@mock.patch("lightning_app.cli.lightning_cli._get_project")
def test_show_logs_errors(project, client):
    """Test that the CLI prints the errors for the show logs command."""

    runner = CliRunner()

    # Response prep
    app = mock.MagicMock()
    app.name = "MyFakeApp"
    work = mock.MagicMock()
    work.name = "MyFakeWork"
    flow = mock.MagicMock()
    flow.name = "MyFakeFlow"

    # No apps ever run
    apps = {}
    client.return_value.lightningapp_instance_service_list_lightningapp_instances.return_value.lightningapps = apps

    result = runner.invoke(logs, ["NonExistentApp"])

    assert result.exit_code == 1
    assert "Error: You don't have any application in the cloud" in result.output

    # App not specified
    apps = {app}
    client.return_value.lightningapp_instance_service_list_lightningapp_instances.return_value.lightningapps = apps

    result = runner.invoke(logs)

    assert result.exit_code == 1
    assert "Please select one of available: [MyFakeApp]" in str(result.output)

    # App does not exit
    apps = {app}
    client.return_value.lightningapp_instance_service_list_lightningapp_instances.return_value.lightningapps = apps

    result = runner.invoke(logs, ["ThisAppDoesNotExist"])

    assert result.exit_code == 1
    assert "The Lightning App 'ThisAppDoesNotExist' does not exist." in str(result.output)

    # Component does not exist
    apps = {app}
    works = {work}
    flows = {flow}
    client.return_value.lightningapp_instance_service_list_lightningapp_instances.return_value.lightningapps = apps
    client.return_value.lightningwork_service_list_lightningwork.return_value.lightningworks = works
    app.spec.flow_servers = flows

    result = runner.invoke(logs, ["MyFakeApp", "NonExistentComponent"])

    assert result.exit_code == 1
    assert "Component 'NonExistentComponent' does not exist in app MyFakeApp." in result.output
