import logging
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
from lightning_cloud.openapi import (
    Body8,
    Gridv1ImageSpec,
    V1BuildSpec,
    V1DependencyFileInfo,
    V1LightningappInstanceState,
    V1LightningworkSpec,
    V1ListLightningappInstancesResponse,
    V1ListMembershipsResponse,
    V1Membership,
    V1NetworkConfig,
    V1PackageManager,
    V1PythonDependencyInfo,
    V1UserRequestedComputeConfig,
    V1Work,
)

from lightning_app import LightningApp, LightningWork
from lightning_app.runners import backends, cloud
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.dependency_caching import get_hash


class MyWork(LightningWork):
    def run(self):
        print("my run")


class TestAppCreationClient:
    """Testing the calls made using GridRestClient to create the app."""

    @mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
    def test_requirements_file(self, monkeypatch):
        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="test-project", project_id="test-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
        monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
        app = mock.MagicMock()
        app.flows = []
        app.frontend = {}
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint_file="entrypoint.py")
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        # without requirements file
        # setting is_file to False so requirements.txt existence check will return False
        monkeypatch.setattr(Path, "is_file", lambda *args, **kwargs: False)
        monkeypatch.setattr(cloud, "Path", Path)
        cloud_runtime.dispatch()
        body = Body8(
            app_entrypoint_file=mock.ANY,
            enable_app_server=True,
            flow_servers=[],
            image_spec=None,
            works=[],
            local_source=True,
            dependency_cache_key=mock.ANY,
        )
        cloud_runtime.backend.client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
            "test-project-id", mock.ANY, body
        )

        # with requirements file
        # setting is_file to True so requirements.txt existence check will return True
        monkeypatch.setattr(Path, "is_file", lambda *args, **kwargs: True)
        monkeypatch.setattr(cloud, "Path", Path)
        cloud_runtime.dispatch()
        body.image_spec = Gridv1ImageSpec(
            dependency_file_info=V1DependencyFileInfo(
                package_manager=V1PackageManager.PIP,
                path="requirements.txt",
            ),
        )
        cloud_runtime.backend.client.lightningapp_v2_service_create_lightningapp_release.assert_called_with(
            "test-project-id", mock.ANY, body
        )

    @mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
    def test_no_cache(self, monkeypatch):
        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="test-project", project_id="test-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
        monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
        monkeypatch.setattr(cloud, "get_hash", lambda *args, **kwargs: "dummy-hash")
        app = mock.MagicMock()
        app.flows = []
        app.frontend = {}
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint_file="entrypoint.py")
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        # requirements.txt check should return True so the no_cache flag is the only one that matters
        # testing with no-cache False
        monkeypatch.setattr(Path, "is_file", lambda self: True if "requirements.txt" in str(self) else False)
        monkeypatch.setattr(cloud, "Path", Path)
        cloud_runtime.dispatch(no_cache=False)
        (
            func_name,
            args,
            kwargs,
        ) = cloud_runtime.backend.client.lightningapp_v2_service_create_lightningapp_release.mock_calls[0]
        body = args[2]
        assert body.dependency_cache_key == "dummy-hash"

        # testing with no-cache True
        mock_client.reset_mock()
        monkeypatch.setattr(cloud, "Path", Path)
        cloud_runtime.dispatch(no_cache=True)
        (
            func_name,
            args,
            kwargs,
        ) = cloud_runtime.backend.client.lightningapp_v2_service_create_lightningapp_release.mock_calls[0]
        body = args[2]
        assert body.dependency_cache_key is None

    @mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
    @pytest.mark.parametrize("lightningapps", [[], [MagicMock()]])
    def test_call_with_work_app(self, lightningapps, monkeypatch, tmpdir):
        source_code_root_dir = Path(tmpdir / "src").absolute()
        source_code_root_dir.mkdir()
        Path(source_code_root_dir / ".lightning").write_text("name: myapp")
        requirements_file = Path(source_code_root_dir / "requirements.txt")
        Path(requirements_file).touch()

        mock_client = mock.MagicMock()
        if lightningapps:
            lightningapps[0].status.phase = V1LightningappInstanceState.STOPPED
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=lightningapps)
        )
        lightning_app_instance = MagicMock()
        mock_client.lightningapp_v2_service_create_lightningapp_release = MagicMock(return_value=lightning_app_instance)
        mock_client.lightningapp_v2_service_create_lightningapp_release_instance = MagicMock(
            return_value=lightning_app_instance
        )
        existing_instance = MagicMock()
        existing_instance.status.phase = V1LightningappInstanceState.STOPPED
        mock_client.lightningapp_service_get_lightningapp = MagicMock(return_value=existing_instance)
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
        monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
        app = mock.MagicMock()
        flow = mock.MagicMock()

        work = MyWork()
        monkeypatch.setattr(work, "_name", "test-work")
        monkeypatch.setattr(work._cloud_build_config, "build_commands", lambda: ["echo 'start'"])
        monkeypatch.setattr(work._cloud_build_config, "requirements", ["torch==1.0.0", "numpy==1.0.0"])
        monkeypatch.setattr(work._cloud_build_config, "image", "random_base_public_image")
        monkeypatch.setattr(work._cloud_compute, "disk_size", 0)
        monkeypatch.setattr(work._cloud_compute, "preemptible", False)
        monkeypatch.setattr(work, "_port", 8080)

        flow.works = lambda recurse: [work]
        app.flows = [flow]
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint_file=(source_code_root_dir / "entrypoint.py"))
        monkeypatch.setattr(
            "lightning_app.runners.cloud._get_project",
            lambda x: V1Membership(name="test-project", project_id="test-project-id"),
        )
        cloud_runtime.dispatch()

        if lightningapps:
            expected_body = Body8(
                description=None,
                local_source=True,
                app_entrypoint_file="entrypoint.py",
                enable_app_server=True,
                flow_servers=[],
                dependency_cache_key=get_hash(requirements_file),
                image_spec=Gridv1ImageSpec(
                    dependency_file_info=V1DependencyFileInfo(
                        package_manager=V1PackageManager.PIP, path="requirements.txt"
                    )
                ),
                works=[
                    V1Work(
                        name="test-work",
                        spec=V1LightningworkSpec(
                            build_spec=V1BuildSpec(
                                commands=["echo 'start'"],
                                python_dependencies=V1PythonDependencyInfo(
                                    package_manager=V1PackageManager.PIP, packages="torch==1.0.0\nnumpy==1.0.0"
                                ),
                                image="random_base_public_image",
                            ),
                            user_requested_compute_config=V1UserRequestedComputeConfig(
                                name="default", count=1, disk_size=0, preemptible=False, shm_size=0
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                        ),
                    )
                ],
            )
            mock_client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
                "test-project-id", mock.ANY, expected_body
            )

            # running dispatch with disabled dependency cache
            mock_client.reset_mock()
            monkeypatch.setattr(cloud, "DISABLE_DEPENDENCY_CACHE", True)
            expected_body.dependency_cache_key = None
            cloud_runtime.dispatch()
            mock_client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
                "test-project-id", mock.ANY, expected_body
            )
        else:
            mock_client.lightningapp_v2_service_create_lightningapp_release_instance.assert_called_once_with(
                "test-project-id", mock.ANY, mock.ANY, mock.ANY
            )


@mock.patch("lightning_app.core.queues.QueuingSystem", MagicMock())
@mock.patch("lightning_app.runners.backends.cloud.LightningClient", MagicMock())
def test_get_project(monkeypatch):
    mock_client = mock.MagicMock()
    monkeypatch.setattr(cloud, "CloudBackend", mock.MagicMock(return_value=mock_client))
    app = mock.MagicMock(spec=LightningApp)
    cloud.CloudRuntime(app=app, entrypoint_file="entrypoint.py")

    # No valid projects
    mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(memberships=[])

    with pytest.raises(ValueError, match="No valid projects found"):
        _get_project(mock_client)

    # One valid project
    mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(name="test-project", project_id="test-project-id")]
    )
    ret = _get_project(mock_client)
    assert ret.project_id == "test-project-id"

    # Multiple valid projects
    mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[
            V1Membership(name="test-project1", project_id="test-project-id1"),
            V1Membership(name="test-project2", project_id="test-project-id2"),
        ]
    )
    with pytest.warns(UserWarning, match="Defaulting to the project test-project1"):
        ret = _get_project(mock_client)
        assert ret.project_id == "test-project-id1"


@mock.patch("lightning_app.core.queues.QueuingSystem", MagicMock())
@mock.patch("lightning_app.runners.backends.cloud.LightningClient", MagicMock())
def test_check_uploaded_folder(monkeypatch, tmpdir, caplog):

    monkeypatch.setattr(cloud, "logger", logging.getLogger())

    app = MagicMock()
    repo = MagicMock()
    backend = cloud.CloudRuntime(app)
    with caplog.at_level(logging.WARN):
        backend._check_uploaded_folder(Path(tmpdir), repo)
    assert caplog.messages == []

    mock = MagicMock()
    mock.st_size = 5 * 1000 * 1000
    repo.files = [str(Path("./a.png"))]
    monkeypatch.setattr(Path, "stat", MagicMock(return_value=mock))

    with caplog.at_level(logging.WARN):
        backend._check_uploaded_folder(Path("."), repo)
    assert caplog.messages[0].startswith("Your application folder . is more than 2 MB. Found 5.0 MB")
