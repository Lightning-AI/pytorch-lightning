import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from lightning_app.cli import cmd_install, lightning_cli
from lightning_app.cli.cmd_install import _install_app
from lightning_app.testing.helpers import RunIf


def test_valid_org_app_name():
    runner = CliRunner()

    # assert a bad app name should fail
    fake_app = "fakeuser/impossible/name"
    result = runner.invoke(lightning_cli.install_app, [fake_app])
    assert "app name format must have organization/app-name" in result.output

    # assert a good name (but unavailable name) should work
    fake_app = "fakeuser/ALKKLJAUHREKJ21234KLAKJDLF"
    result = runner.invoke(lightning_cli.install_app, [fake_app])
    assert f"app: '{fake_app}' is not available on ⚡ Lightning AI ⚡" in result.output
    assert result.exit_code

    # assert a good (and availablea name) works
    real_app = "lightning/install-app"
    result = runner.invoke(lightning_cli.install_app, [real_app])
    assert "Press enter to continue:" in result.output


@pytest.mark.skip(reason="need to figure out how to authorize git clone from the private repo")
def test_valid_unpublished_app_name():
    runner = CliRunner()

    # assert warning of non official app given
    real_app = "https://github.com/PyTorchLightning/install-app"
    try:
        subprocess.check_output(f"lightning install app {real_app}", shell=True, stderr=subprocess.STDOUT)
        # this condition should never be hit
        assert False
    except subprocess.CalledProcessError as e:
        assert "WARNING" in str(e.output)

    # assert aborted install
    result = runner.invoke(lightning_cli.install_app, [real_app], input="q")
    assert "Installation aborted!" in result.output

    # assert a bad app name should fail
    fake_app = "https://github.com/PyTorchLightning/install-appdd"
    result = runner.invoke(lightning_cli.install_app, [fake_app, "--yes"])
    assert "Looks like the github url was not found" in result.output

    # assert a good (and availablea name) works
    result = runner.invoke(lightning_cli.install_app, [real_app])
    assert "Press enter to continue:" in result.output


@pytest.mark.skip(reason="need to figure out how to authorize git clone from the private repo")
def test_app_install(tmpdir):
    """Tests unpublished app install."""

    cwd = os.getcwd()
    os.chdir(tmpdir)

    real_app = "https://github.com/PyTorchLightning/install-app"
    test_app_pip_name = "install-app"

    # install app and verify it's in the env
    subprocess.check_output(f"lightning install app {real_app} --yes", shell=True)
    new_env_output = subprocess.check_output("pip freeze", shell=True)
    assert test_app_pip_name in str(new_env_output), f"{test_app_pip_name} should be in the env"

    os.chdir(cwd)


def test_valid_org_component_name():
    runner = CliRunner()

    # assert a bad name should fail
    fake_component = "fakeuser/impossible/name"
    result = runner.invoke(lightning_cli.install_component, [fake_component])
    assert "component name format must have organization/component-name" in result.output

    # assert a good name (but unavailable name) should work
    fake_component = "fakeuser/ALKKLJAUHREKJ21234KLAKJDLF"
    result = runner.invoke(lightning_cli.install_component, [fake_component])
    assert f"component: '{fake_component}' is not available on ⚡ Lightning AI ⚡" in result.output

    # assert a good (and availablea name) works
    fake_component = "lightning/lit-slack-messenger"
    result = runner.invoke(lightning_cli.install_component, [fake_component])
    assert "Press enter to continue:" in result.output


def test_unpublished_component_url_parsing():
    runner = CliRunner()

    # assert a bad name should fail (no git@)
    fake_component = "https://github.com/PyTorchLightning/LAI-slack-messenger"
    result = runner.invoke(lightning_cli.install_component, [fake_component])
    assert "Error, your github url must be in the following format" in result.output

    # assert a good (and availablea name) works
    sha = "14f333456ffb6758bd19458e6fa0bf12cf5575e1"
    real_component = f"git+https://github.com/PyTorchLightning/LAI-slack-messenger.git@{sha}"
    result = runner.invoke(lightning_cli.install_component, [real_component])
    assert "Press enter to continue:" in result.output


@pytest.mark.skip(reason="need to figure out how to authorize pip install from the private repo")
@pytest.mark.parametrize(
    "real_component, test_component_pip_name",
    [
        ("lightning/lit-slack-messenger", "lit-slack"),
        (
            "git+https://github.com/PyTorchLightning/LAI-slack-messenger.git@14f333456ffb6758bd19458e6fa0bf12cf5575e1",
            "lit-slack",
        ),
    ],
)
def test_component_install(real_component, test_component_pip_name):
    """Tests both published and unpublished component installs."""
    # uninstall component just in case and verify it's not in the pip output
    env_output = subprocess.check_output(f"pip uninstall {test_component_pip_name} --yes && pip freeze", shell=True)
    assert test_component_pip_name not in str(env_output), f"{test_component_pip_name} should not be in the env"

    # install component and verify it's in the env
    new_env_output = subprocess.check_output(
        f"lightning install component {real_component} --yes && pip freeze", shell=True
    )
    assert test_component_pip_name in str(new_env_output), f"{test_component_pip_name} should be in the env"

    # clean up for test
    subprocess.run(f"pip uninstall {test_component_pip_name} --yes", shell=True)
    env_output = subprocess.check_output("pip freeze", shell=True)
    assert test_component_pip_name not in str(
        env_output
    ), f"{test_component_pip_name} should not be in the env after cleanup"


def test_prompt_actions():
    # TODO: each of these installs must check that a package is installed in the environment correctly
    app_to_use = "lightning/install-app"

    runner = CliRunner()

    # assert that the user can cancel the command with any letter other than y
    result = runner.invoke(lightning_cli.install_app, [app_to_use], input="b")
    assert "Installation aborted!" in result.output

    # assert that the install happens with --yes
    # result = runner.invoke(lightning_cli.install_app, [app_to_use, "--yes"])
    # assert result.exit_code == 0

    # assert that the install happens with y
    # result = runner.invoke(lightning_cli.install_app, [app_to_use], input='y')
    # assert result.exit_code == 0

    # # assert that the install happens with yes
    # result = runner.invoke(lightning_cli.install_app, [app_to_use], input='yes')
    # assert result.exit_code == 0

    # assert that the install happens with pressing enter
    # result = runner.invoke(lightning_cli.install_app, [app_to_use])

    # TODO: how to check the output when the user types ctrl+c?
    # result = runner.invoke(lightning_cli.install_app, [app_to_use], input='')


def test_version_arg_component(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    runner = CliRunner()

    # Version does not exist
    component_name = "lightning/lit-slack-messenger"
    version_arg = "NOT-EXIST"
    result = runner.invoke(lightning_cli.install_component, [component_name, f"--version={version_arg}"])
    assert f"component: 'Version {version_arg} for {component_name}' is not" in str(result.exception)
    assert result.exit_code == 1

    # Version exists
    # This somwehow fail in test but not when you actually run it
    version_arg = "0.0.1"
    runner = CliRunner()
    result = runner.invoke(lightning_cli.install_component, [component_name, f"--version={version_arg}", "--yes"])
    assert result.exit_code == 0


def test_version_arg_app(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)

    # Version does not exist
    app_name = "lightning/hackernews-app"
    version_arg = "NOT-EXIST"
    runner = CliRunner()
    result = runner.invoke(lightning_cli.install_app, [app_name, f"--version={version_arg}"])
    assert f"app: 'Version {version_arg} for {app_name}' is not" in str(result.exception)
    assert result.exit_code == 1

    # Version exists
    version_arg = "0.0.1"
    runner = CliRunner()
    result = runner.invoke(lightning_cli.install_app, [app_name, f"--version={version_arg}", "--yes"])
    assert result.exit_code == 0


def test_proper_url_parsing():

    name = "lightning/install-app"

    # make sure org/app-name name is correct
    org, app = cmd_install._validate_name(name, resource_type="app", example="lightning/lit-slack-component")
    assert org == "lightning"
    assert app == "install-app"

    # resolve registry (orgs can have a private registry through their environment variables)
    registry_url = cmd_install._resolve_app_registry()
    assert registry_url == "https://api.sheety.co/e559626ba514c7ba80caae1e38a8d4f4/lightningAppRegistry/apps"

    # load the component resource
    component_entry = cmd_install._resolve_resource(registry_url, name=name, version_arg="latest", resource_type="app")

    source_url, git_url, folder_name = cmd_install._show_install_app_prompt(
        component_entry, app, org, True, resource_type="app"
    )
    assert folder_name == "install-app"
    assert source_url == "https://github.com/PyTorchLightning/install-app"
    assert git_url.find("@") > 10  # TODO: this will be removed once the apps repos will be public
    assert "#ref" not in git_url


@RunIf(skip_windows=True)
def test_install_app_shows_error(tmpdir):

    app_folder_dir = Path(tmpdir / "some_random_directory").absolute()
    app_folder_dir.mkdir()

    with pytest.raises(SystemExit, match=f"Folder {str(app_folder_dir)} exists, please delete it and try again."):
        _install_app(source_url=mock.ANY, git_url=mock.ANY, folder_name=str(app_folder_dir), overwrite=False)


# def test_env_creation(tmpdir):
# cwd = os.getcwd()
# os.chdir(tmpdir)

# # install app
# cmd_install.app("lightning/install-app", True, cwd=tmpdir)

# # assert app folder is installed with venv
# assert "python" in set(os.listdir(os.path.join(tmpdir, "install-app/bin")))

# # assert the deps are in the env
# env_output = subprocess.check_output("source bin/activate && pip freeze", shell=True)
# non_env_output = subprocess.check_output("pip freeze", shell=True)

# # assert envs are not the same
# assert env_output != non_env_output

# # assert the reqs are in the env created and NOT in the non env
# reqs = open(os.path.join(tmpdir, "install-app/requirements.txt")).read()
# assert reqs in str(env_output) and reqs not in str(non_env_output)

# # setup.py installs numpy
# assert "numpy" in str(env_output)

# # run the python script to make sure the file works (in a folder)
# app_file = os.path.join(tmpdir, "install-app/src/app.py")
# app_output = subprocess.check_output(f"source bin/activate && python {app_file}", shell=True)
# assert "b'printed a\\ndeps loaded\\n'" == str(app_output)

# # run the python script to make sure the file works (in root)
# app_file = os.path.join(tmpdir, "install-app/app_b.py")
# app_output = subprocess.check_output(f"source bin/activate && python {app_file}", shell=True)
# assert "b'printed a\\n'" == str(app_output)

# # reset dir
# os.chdir(cwd)


def test_public_app_registry():
    registry = cmd_install._resolve_app_registry()
    assert registry == "https://api.sheety.co/e559626ba514c7ba80caae1e38a8d4f4/lightningAppRegistry/apps"


@mock.patch.dict(os.environ, {"LIGHTNING_APP_REGISTRY": "https://TODO/other_non_PL_registry"})
def test_private_app_registry():
    registry = cmd_install._resolve_app_registry()
    assert registry == "https://TODO/other_non_PL_registry"


def test_public_component_registry():
    registry = cmd_install._resolve_component_registry()
    assert registry == "https://api.sheety.co/e559626ba514c7ba80caae1e38a8d4f4/lightningAppRegistry/components"


@mock.patch.dict(os.environ, {"LIGHTNING_COMPONENT_REGISTRY": "https://TODO/other_non_PL_registry"})
def test_private_component_registry():
    registry = cmd_install._resolve_component_registry()
    assert registry == "https://TODO/other_non_PL_registry"
