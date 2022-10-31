import logging
from copy import copy
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
from lightning_cloud.openapi import (
    Body8,
    Externalv1Cluster,
    Gridv1ImageSpec,
    IdGetBody,
    V1BuildSpec,
    V1DependencyFileInfo,
    V1Drive,
    V1DriveSpec,
    V1DriveStatus,
    V1DriveType,
    V1LightningappInstanceState,
    V1LightningappRelease,
    V1LightningworkDrives,
    V1LightningworkSpec,
    V1ListClustersResponse,
    V1ListLightningappInstancesResponse,
    V1ListMembershipsResponse,
    V1Membership,
    V1Metadata,
    V1NetworkConfig,
    V1PackageManager,
    V1ProjectClusterBinding,
    V1PythonDependencyInfo,
    V1QueueServerType,
    V1SourceType,
    V1UserRequestedComputeConfig,
    V1UserRequestedFlowComputeConfig,
    V1Work,
)

from lightning_app import LightningApp, LightningWork
from lightning_app.runners import backends, cloud
from lightning_app.storage import Drive, Mount
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.dependency_caching import get_hash
from lightning_app.utilities.packaging.cloud_compute import CloudCompute


class MyWork(LightningWork):
    def run(self):
        print("my run")


class WorkWithSingleDrive(LightningWork):
    def __init__(self):
        super().__init__()
        self.drive = None

    def run(self):
        pass


class WorkWithTwoDrives(LightningWork):
    def __init__(self):
        super().__init__()
        self.lit_drive_1 = None
        self.lit_drive_2 = None

    def run(self):
        pass


def get_cloud_runtime_request_body(**kwargs) -> "Body8":
    default_request_body = dict(
        app_entrypoint_file=mock.ANY,
        enable_app_server=True,
        flow_servers=[],
        image_spec=None,
        works=[],
        local_source=True,
        dependency_cache_key=mock.ANY,
        user_requested_flow_compute_config=V1UserRequestedFlowComputeConfig(
            name="flow-lite",
            preemptible=False,
            shm_size=0,
        ),
    )

    if kwargs.get("user_requested_flow_compute_config") is not None:
        default_request_body["user_requested_flow_compute_config"] = kwargs["user_requested_flow_compute_config"]

    return Body8(**default_request_body)


class TestAppCreationClient:
    """Testing the calls made using GridRestClient to create the app."""

    # TODO: remove this test once there is support for multiple instances
    @mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
    def test_new_instance_on_different_cluster(self, monkeypatch):
        app_name = "test-app-name"
        original_cluster = "cluster-001"
        new_cluster = "cluster-002"

        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="Default Project", project_id="default-project-id")]
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease(
            cluster_id=new_cluster
        )
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse(
            [Externalv1Cluster(id=original_cluster), Externalv1Cluster(id=new_cluster)]
        )

        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
        monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))

        app = mock.MagicMock()
        app.flows = []
        app.frontend = {}

        existing_instance = MagicMock()
        existing_instance.status.phase = V1LightningappInstanceState.STOPPED
        existing_instance.spec.cluster_id = original_cluster
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[existing_instance])
        )

        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint_file="entrypoint.py")
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        # without requirements file
        # setting is_file to False so requirements.txt existence check will return False
        monkeypatch.setattr(Path, "is_file", lambda *args, **kwargs: False)
        monkeypatch.setattr(cloud, "Path", Path)

        # This is the main assertion:
        # we have an existing instance on `cluster-001`
        # but we want to run this app on `cluster-002`
        cloud_runtime.dispatch(name=app_name, cluster_id=new_cluster)

        body = Body8(
            cluster_id=new_cluster,
            app_entrypoint_file=mock.ANY,
            enable_app_server=True,
            flow_servers=[],
            image_spec=None,
            works=[],
            local_source=True,
            dependency_cache_key=mock.ANY,
            user_requested_flow_compute_config=mock.ANY,
        )
        cloud_runtime.backend.client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
            project_id="default-project-id", app_id=mock.ANY, body=body
        )
        cloud_runtime.backend.client.projects_service_create_project_cluster_binding.assert_called_once_with(
            project_id="default-project-id",
            body=V1ProjectClusterBinding(cluster_id=new_cluster, project_id="default-project-id"),
        )

    @pytest.mark.parametrize("flow_cloud_compute", [None, CloudCompute(name="t2.medium")])
    @mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
    def test_run_with_default_flow_compute_config(self, monkeypatch, flow_cloud_compute):
        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="test-project", project_id="test-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease(
            cluster_id="test"
        )
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())

        dummy_flow = mock.MagicMock()
        monkeypatch.setattr(dummy_flow, "run", lambda *args, **kwargs: None)
        if flow_cloud_compute is None:
            app = LightningApp(dummy_flow)
        else:
            app = LightningApp(dummy_flow, flow_cloud_compute=flow_cloud_compute)

        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint_file="entrypoint.py")
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        monkeypatch.setattr(Path, "is_file", lambda *args, **kwargs: False)
        monkeypatch.setattr(cloud, "Path", Path)
        cloud_runtime.dispatch()

        user_requested_flow_compute_config = None
        if flow_cloud_compute is not None:
            user_requested_flow_compute_config = V1UserRequestedFlowComputeConfig(
                name=flow_cloud_compute.name,
                preemptible=False,
                shm_size=0,
            )

        body = get_cloud_runtime_request_body(user_requested_flow_compute_config=user_requested_flow_compute_config)
        cloud_runtime.backend.client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
            project_id="test-project-id", app_id=mock.ANY, body=body
        )

    @mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
    def test_run_on_byoc_cluster(self, monkeypatch):
        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="Default Project", project_id="default-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease(
            cluster_id="test1234"
        )
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse(
            [Externalv1Cluster(id="test1234")]
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
        cloud_runtime.dispatch(cluster_id="test1234")
        body = Body8(
            cluster_id="test1234",
            app_entrypoint_file=mock.ANY,
            enable_app_server=True,
            flow_servers=[],
            image_spec=None,
            works=[],
            local_source=True,
            dependency_cache_key=mock.ANY,
            user_requested_flow_compute_config=mock.ANY,
        )
        cloud_runtime.backend.client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
            project_id="default-project-id", app_id=mock.ANY, body=body
        )
        cloud_runtime.backend.client.projects_service_create_project_cluster_binding.assert_called_once_with(
            project_id="default-project-id",
            body=V1ProjectClusterBinding(cluster_id="test1234", project_id="default-project-id"),
        )

    @mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
    def test_requirements_file(self, monkeypatch):
        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="test-project", project_id="test-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease(
            cluster_id="test"
        )
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
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
            user_requested_flow_compute_config=mock.ANY,
        )
        cloud_runtime.backend.client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
            project_id="test-project-id", app_id=mock.ANY, body=body
        )

        # with requirements file
        # setting is_file to True so requirements.txt existence check will return True
        monkeypatch.setattr(Path, "is_file", lambda *args, **kwargs: True)
        monkeypatch.setattr(cloud, "Path", Path)
        cloud_runtime.dispatch(no_cache=True)
        body.image_spec = Gridv1ImageSpec(
            dependency_file_info=V1DependencyFileInfo(
                package_manager=V1PackageManager.PIP,
                path="requirements.txt",
            ),
        )
        body.cluster_id = "test"
        cloud_runtime.backend.client.lightningapp_v2_service_create_lightningapp_release.assert_called_with(
            project_id="test-project-id", app_id=mock.ANY, body=body
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
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease(
            cluster_id="test"
        )
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
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
        body = kwargs["body"]
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
        body = kwargs["body"]
        assert body.dependency_cache_key is None

    @mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
    @pytest.mark.parametrize("lightningapps", [[], [MagicMock()]])
    def test_call_with_work_app(self, lightningapps, monkeypatch, tmpdir):
        source_code_root_dir = Path(tmpdir / "src").absolute()
        source_code_root_dir.mkdir()
        Path(source_code_root_dir / ".lightning").write_text("cluster_id: test\nname: myapp")
        requirements_file = Path(source_code_root_dir / "requirements.txt")
        Path(requirements_file).touch()

        mock_client = mock.MagicMock()
        if lightningapps:
            lightningapps[0].status.phase = V1LightningappInstanceState.STOPPED
            lightningapps[0].spec.cluster_id = "test"
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=lightningapps)
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease(
            cluster_id="test"
        )
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        mock_client.lightningapp_v2_service_create_lightningapp_release_instance.return_value = MagicMock()
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
                user_requested_flow_compute_config=mock.ANY,
                cluster_id="test",
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
                            drives=[],
                            user_requested_compute_config=V1UserRequestedComputeConfig(
                                name="default", count=1, disk_size=0, shm_size=0
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                            cluster_id="test",
                        ),
                    )
                ],
            )
            mock_client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
                project_id="test-project-id", app_id=mock.ANY, body=expected_body
            )

            # running dispatch with disabled dependency cache
            mock_client.reset_mock()
            monkeypatch.setattr(cloud, "DISABLE_DEPENDENCY_CACHE", True)
            expected_body.dependency_cache_key = None
            cloud_runtime.dispatch()
            mock_client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
                project_id="test-project-id", app_id=mock.ANY, body=expected_body
            )
        else:
            mock_client.lightningapp_v2_service_create_lightningapp_release_instance.assert_called_once_with(
                project_id="test-project-id", app_id=mock.ANY, id=mock.ANY, body=mock.ANY
            )

    @mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
    @pytest.mark.parametrize("lightningapps", [[], [MagicMock()]])
    def test_call_with_queue_server_type_specified(self, lightningapps, monkeypatch, tmpdir):
        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="test-project", project_id="test-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease(
            cluster_id="test"
        )
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
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

        # calling with no env variable set
        body = IdGetBody(
            cluster_id="test",
            desired_state=V1LightningappInstanceState.STOPPED,
            env=[],
            name=mock.ANY,
            queue_server_type=V1QueueServerType.UNSPECIFIED,
        )
        client = cloud_runtime.backend.client
        client.lightningapp_v2_service_create_lightningapp_release_instance.assert_called_once_with(
            project_id="test-project-id", app_id=mock.ANY, id=mock.ANY, body=body
        )

        # calling with env variable set to http
        monkeypatch.setattr(cloud, "CLOUD_QUEUE_TYPE", "http")
        cloud_runtime.backend.client.reset_mock()
        cloud_runtime.dispatch()
        body = IdGetBody(
            cluster_id="test",
            desired_state=V1LightningappInstanceState.STOPPED,
            env=[],
            name=mock.ANY,
            queue_server_type=V1QueueServerType.HTTP,
        )
        client = cloud_runtime.backend.client
        client.lightningapp_v2_service_create_lightningapp_release_instance.assert_called_once_with(
            project_id="test-project-id", app_id=mock.ANY, id=mock.ANY, body=body
        )

    @mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
    @pytest.mark.parametrize("lightningapps", [[], [MagicMock()]])
    def test_call_with_work_app_and_attached_drives(self, lightningapps, monkeypatch, tmpdir):
        source_code_root_dir = Path(tmpdir / "src").absolute()
        source_code_root_dir.mkdir()
        Path(source_code_root_dir / ".lightning").write_text("cluster_id: test\nname: myapp")
        requirements_file = Path(source_code_root_dir / "requirements.txt")
        Path(requirements_file).touch()

        mock_client = mock.MagicMock()
        if lightningapps:
            lightningapps[0].status.phase = V1LightningappInstanceState.STOPPED
            lightningapps[0].spec.cluster_id = "test"
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=lightningapps)
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease(
            cluster_id="test"
        )
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        lightning_app_instance = MagicMock()
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

        mocked_drive = MagicMock(spec=Drive)
        setattr(mocked_drive, "id", "foobar")
        setattr(mocked_drive, "protocol", "lit://")
        setattr(mocked_drive, "component_name", "test-work")
        setattr(mocked_drive, "allow_duplicates", False)
        setattr(mocked_drive, "root_folder", tmpdir)
        # deepcopy on a MagicMock instance will return an empty magicmock instance. To
        # overcome this we set the __deepcopy__ method `return_value` to equal what
        # should be the results of the deepcopy operation (an instance of the original class)
        mocked_drive.__deepcopy__.return_value = copy(mocked_drive)

        work = WorkWithSingleDrive()
        monkeypatch.setattr(work, "drive", mocked_drive)
        monkeypatch.setattr(work, "_state", {"_port", "drive"})
        monkeypatch.setattr(work, "_name", "test-work")
        monkeypatch.setattr(work._cloud_build_config, "build_commands", lambda: ["echo 'start'"])
        monkeypatch.setattr(work._cloud_build_config, "requirements", ["torch==1.0.0", "numpy==1.0.0"])
        monkeypatch.setattr(work._cloud_build_config, "image", "random_base_public_image")
        monkeypatch.setattr(work._cloud_compute, "disk_size", 0)
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
                user_requested_flow_compute_config=mock.ANY,
                cluster_id="test",
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
                            drives=[
                                V1LightningworkDrives(
                                    drive=V1Drive(
                                        metadata=V1Metadata(
                                            name="test-work.drive",
                                        ),
                                        spec=V1DriveSpec(
                                            drive_type=V1DriveType.NO_MOUNT_S3,
                                            source_type=V1SourceType.S3,
                                            source="lit://foobar",
                                        ),
                                        status=V1DriveStatus(),
                                    ),
                                    mount_location=str(tmpdir),
                                ),
                            ],
                            user_requested_compute_config=V1UserRequestedComputeConfig(
                                name="default", count=1, disk_size=0, shm_size=0
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                            cluster_id="test",
                        ),
                    )
                ],
            )
            mock_client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
                project_id="test-project-id", app_id=mock.ANY, body=expected_body
            )

            # running dispatch with disabled dependency cache
            mock_client.reset_mock()
            monkeypatch.setattr(cloud, "DISABLE_DEPENDENCY_CACHE", True)
            expected_body.dependency_cache_key = None
            cloud_runtime.dispatch()
            mock_client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
                project_id="test-project-id", app_id=mock.ANY, body=expected_body
            )
        else:
            mock_client.lightningapp_v2_service_create_lightningapp_release_instance.assert_called_once_with(
                project_id="test-project-id", app_id=mock.ANY, id=mock.ANY, body=mock.ANY
            )

    @mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
    @pytest.mark.parametrize("lightningapps", [[], [MagicMock()]])
    def test_call_with_work_app_and_multiple_attached_drives(self, lightningapps, monkeypatch, tmpdir):
        source_code_root_dir = Path(tmpdir / "src").absolute()
        source_code_root_dir.mkdir()
        Path(source_code_root_dir / ".lightning").write_text("cluster_id: test\nname: myapp")
        requirements_file = Path(source_code_root_dir / "requirements.txt")
        Path(requirements_file).touch()

        mock_client = mock.MagicMock()
        if lightningapps:
            lightningapps[0].status.phase = V1LightningappInstanceState.STOPPED
            lightningapps[0].spec.cluster_id = "test"
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=lightningapps)
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease(
            cluster_id="test"
        )
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        lightning_app_instance = MagicMock()
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

        mocked_lit_drive = MagicMock(spec=Drive)
        setattr(mocked_lit_drive, "id", "foobar")
        setattr(mocked_lit_drive, "protocol", "lit://")
        setattr(mocked_lit_drive, "component_name", "test-work")
        setattr(mocked_lit_drive, "allow_duplicates", False)
        setattr(mocked_lit_drive, "root_folder", tmpdir)
        # deepcopy on a MagicMock instance will return an empty magicmock instance. To
        # overcome this we set the __deepcopy__ method `return_value` to equal what
        # should be the results of the deepcopy operation (an instance of the original class)
        mocked_lit_drive.__deepcopy__.return_value = copy(mocked_lit_drive)

        work = WorkWithTwoDrives()
        monkeypatch.setattr(work, "lit_drive_1", mocked_lit_drive)
        monkeypatch.setattr(work, "lit_drive_2", mocked_lit_drive)
        monkeypatch.setattr(work, "_state", {"_port", "_name", "lit_drive_1", "lit_drive_2"})
        monkeypatch.setattr(work, "_name", "test-work")
        monkeypatch.setattr(work._cloud_build_config, "build_commands", lambda: ["echo 'start'"])
        monkeypatch.setattr(work._cloud_build_config, "requirements", ["torch==1.0.0", "numpy==1.0.0"])
        monkeypatch.setattr(work._cloud_build_config, "image", "random_base_public_image")
        monkeypatch.setattr(work._cloud_compute, "disk_size", 0)
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
            lit_drive_1_spec = V1LightningworkDrives(
                drive=V1Drive(
                    metadata=V1Metadata(
                        name="test-work.lit_drive_1",
                    ),
                    spec=V1DriveSpec(
                        drive_type=V1DriveType.NO_MOUNT_S3,
                        source_type=V1SourceType.S3,
                        source="lit://foobar",
                    ),
                    status=V1DriveStatus(),
                ),
                mount_location=str(tmpdir),
            )
            lit_drive_2_spec = V1LightningworkDrives(
                drive=V1Drive(
                    metadata=V1Metadata(
                        name="test-work.lit_drive_2",
                    ),
                    spec=V1DriveSpec(
                        drive_type=V1DriveType.NO_MOUNT_S3,
                        source_type=V1SourceType.S3,
                        source="lit://foobar",
                    ),
                    status=V1DriveStatus(),
                ),
                mount_location=str(tmpdir),
            )

            # order of drives in the spec is non-deterministic, so there are two options
            # depending for the expected body value on which drive is ordered in the list first.

            expected_body_option_1 = Body8(
                description=None,
                local_source=True,
                app_entrypoint_file="entrypoint.py",
                enable_app_server=True,
                flow_servers=[],
                dependency_cache_key=get_hash(requirements_file),
                user_requested_flow_compute_config=mock.ANY,
                cluster_id="test",
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
                            drives=[lit_drive_2_spec, lit_drive_1_spec],
                            user_requested_compute_config=V1UserRequestedComputeConfig(
                                name="default", count=1, disk_size=0, shm_size=0
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                            cluster_id="test",
                        ),
                    )
                ],
            )

            expected_body_option_2 = Body8(
                description=None,
                local_source=True,
                app_entrypoint_file="entrypoint.py",
                enable_app_server=True,
                flow_servers=[],
                dependency_cache_key=get_hash(requirements_file),
                user_requested_flow_compute_config=mock.ANY,
                cluster_id="test",
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
                            drives=[lit_drive_1_spec, lit_drive_2_spec],
                            user_requested_compute_config=V1UserRequestedComputeConfig(
                                name="default", count=1, disk_size=0, shm_size=0
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                            cluster_id="test",
                        ),
                    )
                ],
            )

            # try both options for the expected body to avoid false
            # positive test failures depending on system randomness

            expected_body = expected_body_option_1
            try:
                mock_client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
                    project_id="test-project-id", app_id=mock.ANY, body=expected_body
                )
            except Exception:
                expected_body = expected_body_option_2
                mock_client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
                    project_id="test-project-id", app_id=mock.ANY, body=expected_body
                )

            # running dispatch with disabled dependency cache
            mock_client.reset_mock()
            monkeypatch.setattr(cloud, "DISABLE_DEPENDENCY_CACHE", True)
            expected_body.dependency_cache_key = None
            cloud_runtime.dispatch()
            mock_client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
                project_id="test-project-id", app_id=mock.ANY, body=expected_body
            )
        else:
            mock_client.lightningapp_v2_service_create_lightningapp_release_instance.assert_called_once_with(
                project_id="test-project-id", app_id=mock.ANY, id=mock.ANY, body=mock.ANY
            )

    @mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
    @pytest.mark.parametrize("lightningapps", [[], [MagicMock()]])
    def test_call_with_work_app_and_attached_mount_and_drive(self, lightningapps, monkeypatch, tmpdir):
        source_code_root_dir = Path(tmpdir / "src").absolute()
        source_code_root_dir.mkdir()
        Path(source_code_root_dir / ".lightning").write_text("cluster_id: test\nname: myapp")
        requirements_file = Path(source_code_root_dir / "requirements.txt")
        Path(requirements_file).touch()

        mock_client = mock.MagicMock()
        if lightningapps:
            lightningapps[0].status.phase = V1LightningappInstanceState.STOPPED
            lightningapps[0].spec.cluster_id = "test"
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=lightningapps)
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease(
            cluster_id="test"
        )
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        lightning_app_instance = MagicMock()
        mock_client.lightningapp_v2_service_create_lightningapp_release_instance = MagicMock(
            return_value=lightning_app_instance
        )
        existing_instance = MagicMock()
        existing_instance.status.phase = V1LightningappInstanceState.STOPPED
        existing_instance.spec.cluster_id = None
        mock_client.lightningapp_service_get_lightningapp = MagicMock(return_value=existing_instance)
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
        monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
        app = mock.MagicMock()
        flow = mock.MagicMock()

        mocked_drive = MagicMock(spec=Drive)
        setattr(mocked_drive, "id", "foobar")
        setattr(mocked_drive, "protocol", "lit://")
        setattr(mocked_drive, "component_name", "test-work")
        setattr(mocked_drive, "allow_duplicates", False)
        setattr(mocked_drive, "root_folder", tmpdir)
        # deepcopy on a MagicMock instance will return an empty magicmock instance. To
        # overcome this we set the __deepcopy__ method `return_value` to equal what
        # should be the results of the deepcopy operation (an instance of the original class)
        mocked_drive.__deepcopy__.return_value = copy(mocked_drive)

        mocked_mount = MagicMock(spec=Mount)
        setattr(mocked_mount, "source", "s3://foo/")
        setattr(mocked_mount, "mount_path", "/content/foo")
        setattr(mocked_mount, "protocol", "s3://")

        work = WorkWithSingleDrive()
        monkeypatch.setattr(work, "drive", mocked_drive)
        monkeypatch.setattr(work, "_state", {"_port", "drive"})
        monkeypatch.setattr(work, "_name", "test-work")
        monkeypatch.setattr(work._cloud_build_config, "build_commands", lambda: ["echo 'start'"])
        monkeypatch.setattr(work._cloud_build_config, "requirements", ["torch==1.0.0", "numpy==1.0.0"])
        monkeypatch.setattr(work._cloud_build_config, "image", "random_base_public_image")
        monkeypatch.setattr(work._cloud_compute, "disk_size", 0)
        monkeypatch.setattr(work._cloud_compute, "mounts", mocked_mount)
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
                user_requested_flow_compute_config=mock.ANY,
                cluster_id="test",
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
                            drives=[
                                V1LightningworkDrives(
                                    drive=V1Drive(
                                        metadata=V1Metadata(
                                            name="test-work.drive",
                                        ),
                                        spec=V1DriveSpec(
                                            drive_type=V1DriveType.NO_MOUNT_S3,
                                            source_type=V1SourceType.S3,
                                            source="lit://foobar",
                                        ),
                                        status=V1DriveStatus(),
                                    ),
                                    mount_location=str(tmpdir),
                                ),
                                V1LightningworkDrives(
                                    drive=V1Drive(
                                        metadata=V1Metadata(
                                            name="test-work",
                                        ),
                                        spec=V1DriveSpec(
                                            drive_type=V1DriveType.INDEXED_S3,
                                            source_type=V1SourceType.S3,
                                            source="s3://foo/",
                                        ),
                                        status=V1DriveStatus(),
                                    ),
                                    mount_location="/content/foo",
                                ),
                            ],
                            user_requested_compute_config=V1UserRequestedComputeConfig(
                                name="default", count=1, disk_size=0, shm_size=0
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                            cluster_id="test",
                        ),
                    )
                ],
            )
            mock_client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
                project_id="test-project-id", app_id=mock.ANY, body=expected_body
            )

            # running dispatch with disabled dependency cache
            mock_client.reset_mock()
            monkeypatch.setattr(cloud, "DISABLE_DEPENDENCY_CACHE", True)
            expected_body.dependency_cache_key = None
            cloud_runtime.dispatch()
            mock_client.lightningapp_v2_service_create_lightningapp_release.assert_called_once_with(
                project_id="test-project-id", app_id=mock.ANY, body=expected_body
            )
        else:
            mock_client.lightningapp_v2_service_create_lightningapp_release_instance.assert_called_once_with(
                project_id="test-project-id", app_id=mock.ANY, id=mock.ANY, body=mock.ANY
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


@mock.patch("lightning_app.core.queues.QueuingSystem", MagicMock())
@mock.patch("lightning_app.runners.backends.cloud.LightningClient", MagicMock())
def test_project_has_sufficient_credits():
    app = mock.MagicMock(spec=LightningApp)
    cloud_runtime = cloud.CloudRuntime(app=app, entrypoint_file="entrypoint.py")
    credits_and_test_value = [
        [0.3, False],
        [1, True],
        [1.1, True],
    ]
    for balance, result in credits_and_test_value:
        project = V1Membership(name="test-project1", project_id="test-project-id1", balance=balance)
        assert cloud_runtime._project_has_sufficient_credits(project) is result
