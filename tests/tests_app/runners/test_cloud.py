import contextlib
import logging
import os
import pathlib
import re
import sys
from copy import copy
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
from lightning.app import BuildConfig, LightningApp, LightningFlow, LightningWork
from lightning.app.runners import CloudRuntime, backends, cloud
from lightning.app.source_code.copytree import _copytree, _parse_lightningignore
from lightning.app.source_code.local import LocalSourceCodeDir
from lightning.app.storage import Drive, Mount
from lightning.app.testing.helpers import EmptyWork
from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.dependency_caching import get_hash
from lightning.app.utilities.packaging.cloud_compute import CloudCompute
from lightning_cloud.openapi import (
    CloudspaceIdRunsBody,
    Externalv1Cluster,
    Externalv1LightningappInstance,
    Gridv1ImageSpec,
    IdGetBody,
    ProjectIdProjectclustersbindingsBody,
    V1BuildSpec,
    V1CloudSpace,
    V1CloudSpaceInstanceConfig,
    V1ClusterSpec,
    V1ClusterType,
    V1DataConnectionMount,
    V1DependencyFileInfo,
    V1Drive,
    V1DriveSpec,
    V1DriveStatus,
    V1DriveType,
    V1EnvVar,
    V1GetUserResponse,
    V1LightningappInstanceSpec,
    V1LightningappInstanceState,
    V1LightningappInstanceStatus,
    V1LightningAuth,
    V1LightningBasicAuth,
    V1LightningRun,
    V1LightningworkDrives,
    V1LightningworkSpec,
    V1ListCloudSpacesResponse,
    V1ListClustersResponse,
    V1ListLightningappInstancesResponse,
    V1ListMembershipsResponse,
    V1ListProjectClusterBindingsResponse,
    V1Membership,
    V1Metadata,
    V1NetworkConfig,
    V1PackageManager,
    V1ProjectClusterBinding,
    V1PythonDependencyInfo,
    V1QueueServerType,
    V1SourceType,
    V1UserFeatures,
    V1UserRequestedComputeConfig,
    V1UserRequestedFlowComputeConfig,
    V1Work,
)


class MyWork(LightningWork):
    def run(self):
        print("my run")


class WorkWithSingleDrive(LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drive = None

    def run(self):
        pass


class WorkWithTwoDrives(LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lit_drive_1 = None
        self.lit_drive_2 = None

    def run(self):
        pass


def get_cloud_runtime_request_body(**kwargs) -> "CloudspaceIdRunsBody":
    default_request_body = {
        "app_entrypoint_file": mock.ANY,
        "enable_app_server": True,
        "is_headless": True,
        "should_mount_cloudspace_content": False,
        "flow_servers": [],
        "image_spec": None,
        "works": [],
        "local_source": True,
        "dependency_cache_key": mock.ANY,
        "user_requested_flow_compute_config": V1UserRequestedFlowComputeConfig(
            name="flow-lite",
            preemptible=False,
            shm_size=0,
        ),
    }

    if kwargs.get("user_requested_flow_compute_config") is not None:
        default_request_body["user_requested_flow_compute_config"] = kwargs["user_requested_flow_compute_config"]

    return CloudspaceIdRunsBody(**default_request_body)


@pytest.fixture()
def cloud_backend(monkeypatch):
    cloud_backend = mock.MagicMock()
    monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
    monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
    monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
    return cloud_backend


@pytest.fixture()
def project_id():
    return "test-project-id"


DEFAULT_CLUSTER = "litng-ai-03"


class TestAppCreationClient:
    """Testing the calls made using GridRestClient to create the app."""

    def test_run_on_deleted_cluster(self, cloud_backend):
        app_name = "test-app"

        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="Default Project", project_id=project_id)]
        )

        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([
            Externalv1Cluster(id=DEFAULT_CLUSTER)
        ])
        cloud_backend.client = mock_client

        app = mock.MagicMock()
        app.flows = []
        app.frontend = {}

        existing_instance = MagicMock()
        existing_instance.status.phase = V1LightningappInstanceState.STOPPED
        existing_instance.spec.cluster_id = DEFAULT_CLUSTER
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[existing_instance])
        )

        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=Path("entrypoint.py"))
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        with pytest.raises(ValueError, match="that cluster doesn't exist"):
            cloud_runtime.dispatch(name=app_name, cluster_id="unknown-cluster")

    @pytest.mark.parametrize(
        ("old_cluster", "new_cluster"),
        [
            ("test", "other"),
            ("test", "test"),
            (None, None),
            (None, "litng-ai-03"),
            ("litng-ai-03", None),
        ],
    )
    def test_new_instance_on_different_cluster(self, tmpdir, cloud_backend, project_id, old_cluster, new_cluster):
        entrypoint = Path(tmpdir) / "entrypoint.py"
        entrypoint.touch()

        app_name = "test-app"

        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="Default Project", project_id=project_id)]
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningRun(
            cluster_id=new_cluster
        )

        # Note:
        # backend converts "None" cluster to "litng-ai-03"
        # dispatch should receive None, but API calls should return "litng-ai-03"
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([
            Externalv1Cluster(id=old_cluster or DEFAULT_CLUSTER),
            Externalv1Cluster(id=new_cluster or DEFAULT_CLUSTER),
        ])

        mock_client.projects_service_list_project_cluster_bindings.return_value = V1ListProjectClusterBindingsResponse(
            clusters=[
                V1ProjectClusterBinding(cluster_id=old_cluster or DEFAULT_CLUSTER),
                V1ProjectClusterBinding(cluster_id=new_cluster or DEFAULT_CLUSTER),
            ]
        )

        # Mock all clusters as global clusters
        mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: Externalv1Cluster(
            id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
        )

        cloud_backend.client = mock_client

        app = mock.MagicMock()
        app.flows = []
        app.frontend = {}

        existing_app = MagicMock()
        existing_app.name = app_name
        existing_app.id = "test-id"
        mock_client.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
            cloudspaces=[existing_app]
        )

        existing_instance = MagicMock()
        existing_instance.name = app_name
        existing_instance.status.phase = V1LightningappInstanceState.STOPPED
        existing_instance.spec.cluster_id = old_cluster or DEFAULT_CLUSTER
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[existing_instance])
        )

        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=entrypoint)
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        # This is the main assertion:
        # we have an existing instance on `cluster-001`
        # but we want to run this app on `cluster-002`
        cloud_runtime.dispatch(name=app_name, cluster_id=new_cluster)

        if new_cluster != old_cluster and None not in (old_cluster, new_cluster):
            # If we switched cluster, check that a new name was used which starts with the old name
            mock_client.cloud_space_service_create_lightning_run_instance.assert_called_once()
            args = mock_client.cloud_space_service_create_lightning_run_instance.call_args
            assert args[1]["body"].name != app_name
            assert args[1]["body"].name.startswith(app_name)
            assert args[1]["body"].cluster_id == new_cluster

    def test_running_deleted_app(self, tmpdir, cloud_backend, project_id):
        """Deleted apps show up in list apps but not in list instances.

        This tests that we don't try to reacreate a previously deleted app.

        """
        entrypoint = Path(tmpdir) / "entrypoint.py"
        entrypoint.touch()

        app_name = "test-app"

        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="Default Project", project_id=project_id)]
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningRun(
            cluster_id=DEFAULT_CLUSTER
        )

        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([
            Externalv1Cluster(id=DEFAULT_CLUSTER)
        ])

        mock_client.projects_service_list_project_cluster_bindings.return_value = V1ListProjectClusterBindingsResponse(
            clusters=[V1ProjectClusterBinding(cluster_id=DEFAULT_CLUSTER)]
        )

        # Mock all clusters as global clusters
        mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: Externalv1Cluster(
            id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
        )

        cloud_backend.client = mock_client

        app = mock.MagicMock()
        app.flows = []
        app.frontend = {}

        existing_app = MagicMock()
        existing_app.name = app_name
        existing_app.id = "test-id"
        mock_client.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
            cloudspaces=[existing_app]
        )

        # Simulate the app as deleted so no instance to return
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )

        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=entrypoint)
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        cloud_runtime.dispatch(name=app_name)

        # Check that a new name was used which starts with and does not equal the old name
        mock_client.cloud_space_service_create_lightning_run_instance.assert_called_once()
        args = mock_client.cloud_space_service_create_lightning_run_instance.call_args
        assert args[1]["body"].name != app_name
        assert args[1]["body"].name.startswith(app_name)

    @pytest.mark.parametrize("flow_cloud_compute", [None, CloudCompute(name="t2.medium")])
    @mock.patch("lightning.app.runners.backends.cloud.LightningClient", mock.MagicMock())
    def test_run_with_default_flow_compute_config(self, tmpdir, monkeypatch, flow_cloud_compute):
        entrypoint = Path(tmpdir) / "entrypoint.py"
        entrypoint.touch()

        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="test-project", project_id="test-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningRun(cluster_id="test")
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

        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=entrypoint)
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        cloud_runtime.dispatch()

        user_requested_flow_compute_config = None
        if flow_cloud_compute is not None:
            user_requested_flow_compute_config = V1UserRequestedFlowComputeConfig(
                name=flow_cloud_compute.name, preemptible=False, shm_size=0
            )

        body = get_cloud_runtime_request_body(user_requested_flow_compute_config=user_requested_flow_compute_config)
        cloud_runtime.backend.client.cloud_space_service_create_lightning_run.assert_called_once_with(
            project_id="test-project-id", cloudspace_id=mock.ANY, body=body
        )

    @mock.patch("lightning.app.runners.backends.cloud.LightningClient", mock.MagicMock())
    def test_run_on_byoc_cluster(self, tmpdir, monkeypatch):
        entrypoint = Path(tmpdir) / "entrypoint.py"
        entrypoint.touch()

        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="Default Project", project_id="default-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )
        mock_client.cloud_space_service_create_lightning_run.return_value = V1LightningRun(cluster_id="test1234")
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([
            Externalv1Cluster(id="test1234")
        ])
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
        monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
        app = mock.MagicMock()
        app.is_headless = False
        app.flows = []
        app.frontend = {}
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=entrypoint)
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        cloud_runtime.dispatch(cluster_id="test1234")
        body = CloudspaceIdRunsBody(
            cluster_id="test1234",
            app_entrypoint_file=mock.ANY,
            enable_app_server=True,
            is_headless=False,
            should_mount_cloudspace_content=False,
            flow_servers=[],
            image_spec=None,
            works=[],
            local_source=True,
            dependency_cache_key=mock.ANY,
            user_requested_flow_compute_config=mock.ANY,
        )
        cloud_runtime.backend.client.cloud_space_service_create_lightning_run.assert_called_once_with(
            project_id="default-project-id", cloudspace_id=mock.ANY, body=body
        )
        cloud_runtime.backend.client.projects_service_create_project_cluster_binding.assert_called_once_with(
            project_id="default-project-id",
            body=ProjectIdProjectclustersbindingsBody(cluster_id="test1234"),
        )

    @mock.patch("lightning.app.runners.backends.cloud.LightningClient", mock.MagicMock())
    def test_requirements_file(self, tmpdir, monkeypatch):
        entrypoint = Path(tmpdir) / "entrypoint.py"
        entrypoint.touch()

        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="test-project", project_id="test-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )
        mock_client.cloud_space_service_create_lightning_run.return_value = V1LightningRun()
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
        monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
        app = mock.MagicMock()
        app.is_headless = False
        app.flows = []
        app.frontend = {}
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=entrypoint)
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        # Without requirements file
        cloud_runtime.dispatch()
        body = CloudspaceIdRunsBody(
            app_entrypoint_file=mock.ANY,
            enable_app_server=True,
            is_headless=False,
            should_mount_cloudspace_content=False,
            flow_servers=[],
            image_spec=None,
            works=[],
            local_source=True,
            dependency_cache_key=mock.ANY,
            user_requested_flow_compute_config=mock.ANY,
        )
        cloud_runtime.backend.client.cloud_space_service_create_lightning_run.assert_called_once_with(
            project_id="test-project-id", cloudspace_id=mock.ANY, body=body
        )

        # with requirements file
        requirements = Path(tmpdir) / "requirements.txt"
        requirements.touch()

        cloud_runtime.dispatch(no_cache=True)
        body.image_spec = Gridv1ImageSpec(
            dependency_file_info=V1DependencyFileInfo(package_manager=V1PackageManager.PIP, path="requirements.txt")
        )
        cloud_runtime.backend.client.cloud_space_service_create_lightning_run.assert_called_with(
            project_id="test-project-id", cloudspace_id=mock.ANY, body=body
        )

    @mock.patch("lightning.app.runners.backends.cloud.LightningClient", mock.MagicMock())
    def test_basic_auth_enabled(self, tmpdir, monkeypatch):
        entrypoint = Path(tmpdir) / "entrypoint.py"
        entrypoint.touch()

        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="test-project", project_id="test-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )
        mock_client.cloud_space_service_create_lightning_run.return_value = V1LightningRun()
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
        monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
        app = mock.MagicMock()
        app.is_headless = False
        app.flows = []
        app.frontend = {}
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=entrypoint)
        cloud_runtime._check_uploaded_folder = mock.MagicMock()
        # Set cloud_runtime.enable_basic_auth to be not empty:
        cloud_runtime.enable_basic_auth = "username:password"

        cloud_runtime.dispatch()
        mock_client = cloud_runtime.backend.client

        body = CloudspaceIdRunsBody(
            app_entrypoint_file=mock.ANY,
            enable_app_server=True,
            is_headless=False,
            should_mount_cloudspace_content=False,
            flow_servers=[],
            image_spec=None,
            works=[],
            local_source=True,
            dependency_cache_key=mock.ANY,
            user_requested_flow_compute_config=mock.ANY,
        )

        mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
            project_id="test-project-id", cloudspace_id=mock.ANY, body=body
        )

        mock_client.cloud_space_service_create_lightning_run_instance.assert_called_once_with(
            project_id="test-project-id",
            cloudspace_id=mock.ANY,
            id=mock.ANY,
            body=IdGetBody(
                desired_state=mock.ANY,
                name=mock.ANY,
                env=mock.ANY,
                queue_server_type=mock.ANY,
                auth=V1LightningAuth(basic=V1LightningBasicAuth(username="username", password="password")),
            ),
        )

    @mock.patch("lightning.app.runners.backends.cloud.LightningClient", mock.MagicMock())
    def test_no_cache(self, tmpdir, monkeypatch):
        entrypoint = Path(tmpdir) / "entrypoint.py"
        entrypoint.touch()
        requirements = Path(tmpdir) / "requirements.txt"
        requirements.touch()

        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="test-project", project_id="test-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )
        mock_client.cloud_space_service_create_lightning_run.return_value = V1LightningRun(cluster_id="test")
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
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=entrypoint)
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        # testing with no-cache False
        cloud_runtime.dispatch(no_cache=False)
        _, _, kwargs = cloud_runtime.backend.client.cloud_space_service_create_lightning_run.mock_calls[0]
        body = kwargs["body"]
        assert body.dependency_cache_key == "dummy-hash"

        # testing with no-cache True
        mock_client.reset_mock()
        cloud_runtime.dispatch(no_cache=True)
        _, _, kwargs = cloud_runtime.backend.client.cloud_space_service_create_lightning_run.mock_calls[0]
        body = kwargs["body"]
        assert body.dependency_cache_key is None

    @mock.patch("lightning.app.runners.backends.cloud.LightningClient", mock.MagicMock())
    @pytest.mark.parametrize(
        ("lightningapps", "start_with_flow"),
        [([], False), ([MagicMock()], False), ([MagicMock()], True)],
    )
    def test_call_with_work_app(self, lightningapps, start_with_flow, monkeypatch, tmpdir):
        source_code_root_dir = Path(tmpdir / "src").absolute()
        source_code_root_dir.mkdir()
        Path(source_code_root_dir / ".lightning").write_text("name: myapp")
        requirements_file = Path(source_code_root_dir / "requirements.txt")
        Path(requirements_file).touch()
        (source_code_root_dir / "entrypoint.py").touch()

        mock_client = mock.MagicMock()
        if lightningapps:
            lightningapps[0].name = "myapp"
            lightningapps[0].status.phase = V1LightningappInstanceState.STOPPED
            lightningapps[0].spec.cluster_id = "test"
        mock_client.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
            cloudspaces=lightningapps
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=lightningapps)
        )
        mock_client.projects_service_list_project_cluster_bindings.return_value = V1ListProjectClusterBindingsResponse(
            clusters=[V1ProjectClusterBinding(cluster_id="test")]
        )
        mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: Externalv1Cluster(
            id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
        )
        mock_client.cloud_space_service_create_lightning_run_instance.return_value = V1LightningRun()
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        mock_client.cloud_space_service_create_lightning_run_instance.return_value = MagicMock()
        existing_instance = MagicMock()
        existing_instance.status.phase = V1LightningappInstanceState.STOPPED
        mock_client.lightningapp_service_get_lightningapp = MagicMock(return_value=existing_instance)
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
        monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
        app = mock.MagicMock()
        app.is_headless = False

        work = MyWork(start_with_flow=start_with_flow, cloud_compute=CloudCompute("custom"))
        work._name = "test-work"
        work._cloud_build_config.build_commands = lambda: ["echo 'start'"]
        work._cloud_build_config.requirements = ["torch==1.0.0", "numpy==1.0.0"]
        work._cloud_build_config.image = "random_base_public_image"
        work._cloud_compute.disk_size = 0
        work._port = 8080

        app.works = [work]
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=(source_code_root_dir / "entrypoint.py"))
        monkeypatch.setattr(
            "lightning.app.runners.cloud._get_project",
            lambda _, project_id: V1Membership(name="test-project", project_id="test-project-id"),
        )
        cloud_runtime.dispatch()

        if lightningapps:
            expected_body = CloudspaceIdRunsBody(
                description=None,
                local_source=True,
                app_entrypoint_file="entrypoint.py",
                enable_app_server=True,
                is_headless=False,
                should_mount_cloudspace_content=False,
                flow_servers=[],
                dependency_cache_key=get_hash(requirements_file),
                user_requested_flow_compute_config=mock.ANY,
                cluster_id="test",
                image_spec=Gridv1ImageSpec(
                    dependency_file_info=V1DependencyFileInfo(
                        package_manager=V1PackageManager.PIP, path="requirements.txt"
                    )
                ),
            )

            if start_with_flow:
                expected_body.works = [
                    V1Work(
                        name="test-work",
                        display_name="",
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
                                name="custom",
                                count=1,
                                disk_size=0,
                                shm_size=0,
                                preemptible=False,
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                            data_connection_mounts=[],
                        ),
                    )
                ]
            else:
                expected_body.works = []

            mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
                project_id="test-project-id", cloudspace_id=mock.ANY, body=expected_body
            )

            # running dispatch with disabled dependency cache
            mock_client.reset_mock()
            monkeypatch.setattr(cloud, "DISABLE_DEPENDENCY_CACHE", True)
            expected_body.dependency_cache_key = None
            cloud_runtime.dispatch()
            mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
                project_id="test-project-id", cloudspace_id=mock.ANY, body=expected_body
            )
        else:
            mock_client.cloud_space_service_create_lightning_run_instance.assert_called_once_with(
                project_id="test-project-id", cloudspace_id=mock.ANY, id=mock.ANY, body=mock.ANY
            )

    @mock.patch("lightning.app.runners.backends.cloud.LightningClient", mock.MagicMock())
    @pytest.mark.parametrize("lightningapps", [[], [MagicMock()]])
    def test_call_with_queue_server_type_specified(self, tmpdir, lightningapps, monkeypatch):
        entrypoint = Path(tmpdir) / "entrypoint.py"
        entrypoint.touch()

        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="test-project", project_id="test-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )
        mock_client.cloud_space_service_create_lightning_run.return_value = V1LightningRun()
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
        monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
        app = mock.MagicMock()
        app.flows = []
        app.frontend = {}
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=entrypoint)
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        cloud_runtime.dispatch()

        # calling with no env variable set
        body = IdGetBody(
            desired_state=V1LightningappInstanceState.STOPPED,
            env=[],
            name=mock.ANY,
            queue_server_type=V1QueueServerType.UNSPECIFIED,
        )
        client = cloud_runtime.backend.client
        client.cloud_space_service_create_lightning_run_instance.assert_called_once_with(
            project_id="test-project-id", cloudspace_id=mock.ANY, id=mock.ANY, body=body
        )

        # calling with env variable set to http
        monkeypatch.setitem(os.environ, "LIGHTNING_CLOUD_QUEUE_TYPE", "http")
        cloud_runtime.backend.client.reset_mock()
        cloud_runtime.dispatch()
        body = IdGetBody(
            desired_state=V1LightningappInstanceState.STOPPED,
            env=mock.ANY,
            name=mock.ANY,
            queue_server_type=V1QueueServerType.HTTP,
        )
        client = cloud_runtime.backend.client
        client.cloud_space_service_create_lightning_run_instance.assert_called_once_with(
            project_id="test-project-id", cloudspace_id=mock.ANY, id=mock.ANY, body=body
        )

    @mock.patch("lightning.app.runners.backends.cloud.LightningClient", mock.MagicMock())
    @pytest.mark.parametrize("lightningapps", [[], [MagicMock()]])
    def test_call_with_work_app_and_attached_drives(self, lightningapps, monkeypatch, tmpdir):
        source_code_root_dir = Path(tmpdir / "src").absolute()
        source_code_root_dir.mkdir()
        Path(source_code_root_dir / ".lightning").write_text("name: myapp")
        requirements_file = Path(source_code_root_dir / "requirements.txt")
        Path(requirements_file).touch()
        (source_code_root_dir / "entrypoint.py").touch()

        mock_client = mock.MagicMock()
        if lightningapps:
            lightningapps[0].name = "myapp"
            lightningapps[0].status.phase = V1LightningappInstanceState.STOPPED
            lightningapps[0].spec.cluster_id = "test"
        mock_client.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
            cloudspaces=lightningapps
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=lightningapps)
        )
        mock_client.projects_service_list_project_cluster_bindings.return_value = V1ListProjectClusterBindingsResponse(
            clusters=[V1ProjectClusterBinding(cluster_id="test")]
        )
        mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: Externalv1Cluster(
            id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
        )
        mock_client.cloud_space_service_create_lightning_run_instance.return_value = V1LightningRun()
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        lit_app_instance = MagicMock()
        mock_client.cloud_space_service_create_lightning_run_instance = MagicMock(return_value=lit_app_instance)
        existing_instance = MagicMock()
        existing_instance.status.phase = V1LightningappInstanceState.STOPPED
        mock_client.lightningapp_service_get_lightningapp = MagicMock(return_value=existing_instance)
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
        monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
        app = mock.MagicMock()
        app.is_headless = False

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

        work = WorkWithSingleDrive(cloud_compute=CloudCompute("custom"))
        monkeypatch.setattr(work, "drive", mocked_drive)
        monkeypatch.setattr(work, "_state", {"_port", "drive"})
        monkeypatch.setattr(work, "_name", "test-work")
        monkeypatch.setattr(work._cloud_build_config, "build_commands", lambda: ["echo 'start'"])
        monkeypatch.setattr(work._cloud_build_config, "requirements", ["torch==1.0.0", "numpy==1.0.0"])
        monkeypatch.setattr(work._cloud_build_config, "image", "random_base_public_image")
        monkeypatch.setattr(work._cloud_compute, "disk_size", 0)
        monkeypatch.setattr(work, "_port", 8080)

        app.works = [work]
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=(source_code_root_dir / "entrypoint.py"))
        monkeypatch.setattr(
            "lightning.app.runners.cloud._get_project",
            lambda _, project_id: V1Membership(name="test-project", project_id="test-project-id"),
        )
        cloud_runtime.dispatch()

        if lightningapps:
            expected_body = CloudspaceIdRunsBody(
                description=None,
                local_source=True,
                app_entrypoint_file="entrypoint.py",
                enable_app_server=True,
                is_headless=False,
                should_mount_cloudspace_content=False,
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
                        display_name="",
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
                                ),
                            ],
                            user_requested_compute_config=V1UserRequestedComputeConfig(
                                name="custom", count=1, disk_size=0, shm_size=0, preemptible=False
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                            data_connection_mounts=[],
                        ),
                    )
                ],
            )
            mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
                project_id="test-project-id", cloudspace_id=mock.ANY, body=expected_body
            )

            # running dispatch with disabled dependency cache
            mock_client.reset_mock()
            monkeypatch.setattr(cloud, "DISABLE_DEPENDENCY_CACHE", True)
            expected_body.dependency_cache_key = None
            cloud_runtime.dispatch()
            mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
                project_id="test-project-id", cloudspace_id=mock.ANY, body=expected_body
            )
        else:
            mock_client.cloud_space_service_create_lightning_run_instance.assert_called_once_with(
                project_id="test-project-id", cloudspace_id=mock.ANY, id=mock.ANY, body=mock.ANY
            )

    @mock.patch("lightning.app.runners.backends.cloud.LightningClient", mock.MagicMock())
    @mock.patch("lightning.app.core.constants.ENABLE_APP_COMMENT_COMMAND_EXECUTION", True)
    @pytest.mark.parametrize("lightningapps", [[], [MagicMock()]])
    def test_call_with_work_app_and_app_comment_command_execution_set(self, lightningapps, monkeypatch, tmpdir):
        source_code_root_dir = Path(tmpdir / "src").absolute()
        source_code_root_dir.mkdir()
        Path(source_code_root_dir / ".lightning").write_text("name: myapp")
        requirements_file = Path(source_code_root_dir / "requirements.txt")
        Path(requirements_file).touch()
        (source_code_root_dir / "entrypoint.py").touch()

        mock_client = mock.MagicMock()
        if lightningapps:
            lightningapps[0].name = "myapp"
            lightningapps[0].status.phase = V1LightningappInstanceState.STOPPED
            lightningapps[0].spec.cluster_id = "test"
            mock_client.projects_service_list_project_cluster_bindings.return_value = (
                V1ListProjectClusterBindingsResponse(clusters=[V1ProjectClusterBinding(cluster_id="test")])
            )
            mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: Externalv1Cluster(
                id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
            )
        mock_client.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
            cloudspaces=lightningapps
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=lightningapps)
        )
        mock_client.cloud_space_service_create_lightning_run_instance.return_value = V1LightningRun()
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        lit_app_instance = MagicMock()
        mock_client.cloud_space_service_create_lightning_run_instance = MagicMock(return_value=lit_app_instance)
        existing_instance = MagicMock()
        existing_instance.status.phase = V1LightningappInstanceState.STOPPED
        mock_client.lightningapp_service_get_lightningapp = MagicMock(return_value=existing_instance)
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
        monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
        app = mock.MagicMock()
        app.is_headless = False

        work = MyWork(cloud_compute=CloudCompute("custom"))
        work._state = {"_port"}
        work._name = "test-work"
        work._cloud_build_config.build_commands = lambda: ["echo 'start'"]
        work._cloud_build_config.requirements = ["torch==1.0.0", "numpy==1.0.0"]
        work._cloud_build_config.image = "random_base_public_image"
        work._cloud_compute.disk_size = 0
        work._port = 8080

        app.works = [work]
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=(source_code_root_dir / "entrypoint.py"))
        monkeypatch.setattr(
            "lightning.app.runners.cloud._get_project",
            lambda _, project_id: V1Membership(name="test-project", project_id="test-project-id"),
        )
        cloud_runtime.run_app_comment_commands = True
        cloud_runtime.dispatch()

        if lightningapps:
            expected_body = CloudspaceIdRunsBody(
                description=None,
                local_source=True,
                app_entrypoint_file="entrypoint.py",
                enable_app_server=True,
                is_headless=False,
                should_mount_cloudspace_content=False,
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
                        display_name="",
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
                                name="custom", count=1, disk_size=0, shm_size=0, preemptible=mock.ANY
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                            cluster_id=mock.ANY,
                            data_connection_mounts=[],
                        ),
                    )
                ],
            )

            mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
                project_id="test-project-id", cloudspace_id=mock.ANY, body=expected_body
            )

            # running dispatch with disabled dependency cache
            mock_client.reset_mock()
            monkeypatch.setattr(cloud, "DISABLE_DEPENDENCY_CACHE", True)
            expected_body.dependency_cache_key = None
            cloud_runtime.dispatch()
            mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
                project_id="test-project-id", cloudspace_id=mock.ANY, body=expected_body
            )
        else:
            mock_client.cloud_space_service_create_lightning_run_instance.assert_called_once_with(
                project_id="test-project-id",
                cloudspace_id=mock.ANY,
                id=mock.ANY,
                body=IdGetBody(
                    desired_state=V1LightningappInstanceState.STOPPED,
                    name=mock.ANY,
                    env=[V1EnvVar(name="ENABLE_APP_COMMENT_COMMAND_EXECUTION", value="1")],
                    queue_server_type=mock.ANY,
                ),
            )

    @mock.patch("lightning.app.runners.backends.cloud.LightningClient", mock.MagicMock())
    @pytest.mark.parametrize("lightningapps", [[], [MagicMock()]])
    def test_call_with_work_app_and_multiple_attached_drives(self, lightningapps, monkeypatch, tmpdir):
        source_code_root_dir = Path(tmpdir / "src").absolute()
        source_code_root_dir.mkdir()
        Path(source_code_root_dir / ".lightning").write_text("name: myapp")
        requirements_file = Path(source_code_root_dir / "requirements.txt")
        Path(requirements_file).touch()
        (source_code_root_dir / "entrypoint.py").touch()

        mock_client = mock.MagicMock()
        if lightningapps:
            lightningapps[0].name = "myapp"
            lightningapps[0].status.phase = V1LightningappInstanceState.STOPPED
            lightningapps[0].spec.cluster_id = "test"
            mock_client.projects_service_list_project_cluster_bindings.return_value = (
                V1ListProjectClusterBindingsResponse(
                    clusters=[
                        V1ProjectClusterBinding(cluster_id="test"),
                    ]
                )
            )
            mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: Externalv1Cluster(
                id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
            )
        mock_client.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
            cloudspaces=lightningapps
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=lightningapps)
        )
        mock_client.cloud_space_service_create_lightning_run_instance.return_value = V1LightningRun(cluster_id="test")
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        lit_app_instance = MagicMock()
        mock_client.cloud_space_service_create_lightning_run_instance = MagicMock(return_value=lit_app_instance)
        existing_instance = MagicMock()
        existing_instance.status.phase = V1LightningappInstanceState.STOPPED
        mock_client.lightningapp_service_get_lightningapp = MagicMock(return_value=existing_instance)
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
        monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
        app = mock.MagicMock()
        app.is_headless = False

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

        work = WorkWithTwoDrives(cloud_compute=CloudCompute("custom"))
        work.lit_drive_1 = mocked_lit_drive
        work.lit_drive_2 = mocked_lit_drive
        work._state = {"_port", "_name", "lit_drive_1", "lit_drive_2"}
        work._name = "test-work"
        work._cloud_build_config.build_commands = lambda: ["echo 'start'"]
        work._cloud_build_config.requirements = ["torch==1.0.0", "numpy==1.0.0"]
        work._cloud_build_config.image = "random_base_public_image"
        work._cloud_compute.disk_size = 0
        work._port = 8080

        app.works = [work]
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=(source_code_root_dir / "entrypoint.py"))
        monkeypatch.setattr(
            "lightning.app.runners.cloud._get_project",
            lambda _, project_id: V1Membership(name="test-project", project_id="test-project-id"),
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
            )

            # order of drives in the spec is non-deterministic, so there are two options
            # depending for the expected body value on which drive is ordered in the list first.

            expected_body_option_1 = CloudspaceIdRunsBody(
                description=None,
                local_source=True,
                app_entrypoint_file="entrypoint.py",
                enable_app_server=True,
                is_headless=False,
                should_mount_cloudspace_content=False,
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
                        display_name="",
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
                                name="custom",
                                count=1,
                                disk_size=0,
                                shm_size=0,
                                preemptible=False,
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                            data_connection_mounts=[],
                        ),
                    )
                ],
            )

            expected_body_option_2 = CloudspaceIdRunsBody(
                description=None,
                local_source=True,
                app_entrypoint_file="entrypoint.py",
                enable_app_server=True,
                is_headless=False,
                should_mount_cloudspace_content=False,
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
                        display_name="",
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
                                name="custom",
                                count=1,
                                disk_size=0,
                                shm_size=0,
                                preemptible=False,
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                            data_connection_mounts=[],
                        ),
                    )
                ],
            )

            # try both options for the expected body to avoid false
            # positive test failures depending on system randomness

            expected_body = expected_body_option_1
            try:
                mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
                    project_id="test-project-id", cloudspace_id=mock.ANY, body=expected_body
                )
            except Exception:
                expected_body = expected_body_option_2
                mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
                    project_id="test-project-id", cloudspace_id=mock.ANY, body=expected_body
                )

            # running dispatch with disabled dependency cache
            mock_client.reset_mock()
            monkeypatch.setattr(cloud, "DISABLE_DEPENDENCY_CACHE", True)
            expected_body.dependency_cache_key = None
            cloud_runtime.dispatch()
            mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
                project_id="test-project-id", cloudspace_id=mock.ANY, body=expected_body
            )
        else:
            mock_client.cloud_space_service_create_lightning_run_instance.assert_called_once_with(
                project_id="test-project-id", cloudspace_id=mock.ANY, id=mock.ANY, body=mock.ANY
            )

    @mock.patch("lightning.app.runners.backends.cloud.LightningClient", mock.MagicMock())
    @pytest.mark.parametrize("lightningapps", [[], [MagicMock()]])
    def test_call_with_work_app_and_attached_mount_and_drive(self, lightningapps, monkeypatch, tmpdir):
        source_code_root_dir = Path(tmpdir / "src").absolute()
        source_code_root_dir.mkdir()
        Path(source_code_root_dir / ".lightning").write_text("name: myapp")
        requirements_file = Path(source_code_root_dir / "requirements.txt")
        Path(requirements_file).touch()
        (source_code_root_dir / "entrypoint.py").touch()

        mock_client = mock.MagicMock()
        if lightningapps:
            lightningapps[0].name = "myapp"
            lightningapps[0].status.phase = V1LightningappInstanceState.STOPPED
            lightningapps[0].spec.cluster_id = "test"
            mock_client.projects_service_list_project_cluster_bindings.return_value = (
                V1ListProjectClusterBindingsResponse(clusters=[V1ProjectClusterBinding(cluster_id="test")])
            )
            mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: Externalv1Cluster(
                id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
            )
        mock_client.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
            cloudspaces=lightningapps
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=lightningapps)
        )
        mock_client.cloud_space_service_create_lightning_run_instance.return_value = V1LightningRun(cluster_id="test")
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        lit_app_instance = MagicMock()
        mock_client.cloud_space_service_create_lightning_run_instance = MagicMock(return_value=lit_app_instance)
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
        app.is_headless = False

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

        work = WorkWithSingleDrive(cloud_compute=CloudCompute("custom"))
        monkeypatch.setattr(work, "drive", mocked_drive)
        monkeypatch.setattr(work, "_state", {"_port", "drive"})
        monkeypatch.setattr(work, "_name", "test-work")
        monkeypatch.setattr(work._cloud_build_config, "build_commands", lambda: ["echo 'start'"])
        monkeypatch.setattr(work._cloud_build_config, "requirements", ["torch==1.0.0", "numpy==1.0.0"])
        monkeypatch.setattr(work._cloud_build_config, "image", "random_base_public_image")
        monkeypatch.setattr(work._cloud_compute, "disk_size", 0)
        monkeypatch.setattr(work._cloud_compute, "mounts", mocked_mount)
        monkeypatch.setattr(work, "_port", 8080)

        app.works = [work]
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=(source_code_root_dir / "entrypoint.py"))
        monkeypatch.setattr(
            "lightning.app.runners.cloud._get_project",
            lambda _, project_id: V1Membership(name="test-project", project_id="test-project-id"),
        )
        cloud_runtime.dispatch()

        if lightningapps:
            expected_body = CloudspaceIdRunsBody(
                description=None,
                local_source=True,
                app_entrypoint_file="entrypoint.py",
                enable_app_server=True,
                is_headless=False,
                should_mount_cloudspace_content=False,
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
                        display_name="",
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
                                name="custom",
                                count=1,
                                disk_size=0,
                                shm_size=0,
                                preemptible=False,
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                            data_connection_mounts=[],
                        ),
                    )
                ],
            )
            mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
                project_id="test-project-id", cloudspace_id=mock.ANY, body=expected_body
            )

            # running dispatch with disabled dependency cache
            mock_client.reset_mock()
            monkeypatch.setattr(cloud, "DISABLE_DEPENDENCY_CACHE", True)
            expected_body.dependency_cache_key = None
            cloud_runtime.dispatch()
            mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
                project_id="test-project-id", cloudspace_id=mock.ANY, body=expected_body
            )
        else:
            mock_client.cloud_space_service_create_lightning_run_instance.assert_called_once_with(
                project_id="test-project-id", cloudspace_id=mock.ANY, id=mock.ANY, body=mock.ANY
            )


class TestOpen:
    def test_open(self, monkeypatch):
        """Tests that the open method calls the expected API endpoints."""
        mock_client = mock.MagicMock()
        mock_client.auth_service_get_user.return_value = V1GetUserResponse(
            username="tester", features=V1UserFeatures(code_tab=True)
        )
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="test-project", project_id="test-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )

        mock_client.cloud_space_service_create_cloud_space.return_value = V1CloudSpace(id="cloudspace_id")
        mock_client.cloud_space_service_create_lightning_run.return_value = V1LightningRun(id="run_id")

        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        mock_local_source = mock.MagicMock()
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock_local_source)

        cloud_runtime = cloud.CloudRuntime(entrypoint=Path("."))

        cloud_runtime.open("test_space")

        mock_client.cloud_space_service_create_cloud_space.assert_called_once_with(
            project_id="test-project-id", body=mock.ANY
        )
        mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
            project_id="test-project-id", cloudspace_id="cloudspace_id", body=mock.ANY
        )

        assert mock_client.cloud_space_service_create_cloud_space.call_args.kwargs["body"].name == "test_space"

    @pytest.mark.parametrize(
        ("path", "expected_root", "entries", "expected_filtered_entries"),
        [(".", ".", ["a.py", "b.ipynb"], ["a.py", "b.ipynb"]), ("a.py", ".", ["a.py", "b.ipynb"], ["a.py"])],
    )
    def test_open_repo(self, tmpdir, monkeypatch, path, expected_root, entries, expected_filtered_entries):
        """Tests that the local source code repo is set up with the correct path and ignore functions."""
        tmpdir = Path(tmpdir)
        for entry in entries:
            (tmpdir / entry).touch()

        mock_client = mock.MagicMock()
        mock_client.auth_service_get_user.return_value = V1GetUserResponse(
            username="tester", features=V1UserFeatures(code_tab=True)
        )
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="test-project", project_id="test-project-id")]
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningRun(cluster_id="test")
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([Externalv1Cluster(id="test")])
        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        mock_local_source = mock.MagicMock()
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock_local_source)

        cloud_runtime = cloud.CloudRuntime(entrypoint=tmpdir / path)

        cloud_runtime.open("test_space")

        mock_local_source.assert_called_once()
        repo_call = mock_local_source.call_args

        assert repo_call.kwargs["path"] == (tmpdir / expected_root).absolute()
        ignore_functions = repo_call.kwargs["ignore_functions"]
        if len(ignore_functions) > 0:
            filtered = ignore_functions[0]("", [tmpdir / entry for entry in entries])
        else:
            filtered = [tmpdir / entry for entry in entries]

        filtered = [entry.absolute() for entry in filtered]
        expected_filtered_entries = [(tmpdir / entry).absolute() for entry in expected_filtered_entries]
        assert filtered == expected_filtered_entries

    def test_reopen(self, monkeypatch, capsys):
        """Tests that the open method calls the expected API endpoints when the CloudSpace already exists."""
        mock_client = mock.MagicMock()
        mock_client.auth_service_get_user.return_value = V1GetUserResponse(
            username="tester", features=V1UserFeatures(code_tab=True)
        )
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="test-project", project_id="test-project-id")]
        )

        mock_client.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
            cloudspaces=[V1CloudSpace(id="cloudspace_id", name="test_space")]
        )

        running_instance = Externalv1LightningappInstance(
            id="instance_id",
            name="test_space",
            spec=V1LightningappInstanceSpec(cluster_id="test"),
            status=V1LightningappInstanceStatus(phase=V1LightningappInstanceState.RUNNING),
        )

        stopped_instance = Externalv1LightningappInstance(
            id="instance_id",
            name="test_space",
            spec=V1LightningappInstanceSpec(cluster_id="test"),
            status=V1LightningappInstanceStatus(phase=V1LightningappInstanceState.STOPPED),
        )

        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[running_instance])
        )
        mock_client.lightningapp_instance_service_update_lightningapp_instance.return_value = running_instance
        mock_client.lightningapp_instance_service_get_lightningapp_instance.return_value = stopped_instance

        mock_client.cloud_space_service_create_cloud_space.return_value = V1CloudSpace(id="cloudspace_id")
        mock_client.cloud_space_service_create_lightning_run.return_value = V1LightningRun(id="run_id")

        cluster = Externalv1Cluster(id="test", spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL))
        mock_client.projects_service_list_project_cluster_bindings.return_value = V1ListProjectClusterBindingsResponse(
            clusters=[V1ProjectClusterBinding(cluster_id="test")],
        )
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([cluster])
        mock_client.cluster_service_get_cluster.return_value = cluster

        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        mock_local_source = mock.MagicMock()
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock_local_source)

        cloud_runtime = cloud.CloudRuntime(entrypoint=Path("."))

        cloud_runtime.open("test_space")

        mock_client.cloud_space_service_create_lightning_run_instance.assert_not_called()
        mock_client.cloud_space_service_create_cloud_space.assert_not_called()

        mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
            project_id="test-project-id", cloudspace_id="cloudspace_id", body=mock.ANY
        )

    def test_not_enabled(self, monkeypatch, capsys):
        """Tests that an error is printed and the call exits if the feature isn't enabled for the user."""
        mock_client = mock.MagicMock()
        mock_client.auth_service_get_user.return_value = V1GetUserResponse(
            username="tester",
            features=V1UserFeatures(code_tab=False),
        )

        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))

        cloud_runtime = cloud.CloudRuntime(entrypoint=Path("."))

        monkeypatch.setattr(cloud, "Path", Path)

        exited = False
        try:
            cloud_runtime.open("test_space")
        except SystemExit:
            # Expected behaviour
            exited = True

        out, _ = capsys.readouterr()

        assert exited
        assert "`lightning_app open` command has not been enabled" in out


class TestCloudspaceDispatch:
    @mock.patch.object(pathlib.Path, "exists")
    @pytest.mark.parametrize(
        ("custom_env_sync_path_value", "cloudspace"),
        [
            (None, V1CloudSpace(id="test_id", code_config=V1CloudSpaceInstanceConfig())),
            (
                Path("/tmp/sys-customizations-sync"),
                V1CloudSpace(id="test_id", code_config=V1CloudSpaceInstanceConfig()),
            ),
            (
                Path("/tmp/sys-customizations-sync"),
                V1CloudSpace(
                    id="test_id",
                    code_config=V1CloudSpaceInstanceConfig(data_connection_mounts=[V1DataConnectionMount(id="test")]),
                ),
            ),
        ],
    )
    def test_cloudspace_dispatch(self, custom_env_sync_root, custom_env_sync_path_value, cloudspace, monkeypatch):
        """Tests that the cloudspace_dispatch method calls the expected API endpoints."""
        mock_client = mock.MagicMock()
        mock_client.auth_service_get_user.return_value = V1GetUserResponse(
            username="tester",
            features=V1UserFeatures(),
        )
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="project", project_id="project_id")]
        )
        mock_client.cloud_space_service_create_lightning_run.return_value = V1LightningRun(id="run_id")
        mock_client.cloud_space_service_create_lightning_run_instance.return_value = Externalv1LightningappInstance(
            id="instance_id"
        )

        cluster = Externalv1Cluster(id="test", spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL))
        mock_client.projects_service_list_project_cluster_bindings.return_value = V1ListProjectClusterBindingsResponse(
            clusters=[V1ProjectClusterBinding(cluster_id="cluster_id")],
        )
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse([cluster])
        mock_client.cluster_service_get_cluster.return_value = cluster
        mock_client.cloud_space_service_get_cloud_space.return_value = cloudspace

        cloud_backend = mock.MagicMock()
        cloud_backend.client = mock_client
        monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
        mock_repo = mock.MagicMock()
        mock_local_source = mock.MagicMock(return_value=mock_repo)
        monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock_local_source)
        custom_env_sync_root.return_value = custom_env_sync_path_value

        mock_app = mock.MagicMock()
        mock_app.works = [mock.MagicMock()]
        cloud_runtime = cloud.CloudRuntime(app=mock_app, entrypoint=Path("."))

        app = cloud_runtime.cloudspace_dispatch("project_id", "cloudspace_id", "run_name", "cluster_id")
        assert app.id == "instance_id"

        mock_client.cloud_space_service_get_cloud_space.assert_called_once_with(
            project_id="project_id", id="cloudspace_id"
        )

        mock_client.cloud_space_service_create_lightning_run.assert_called_once_with(
            project_id="project_id", cloudspace_id="cloudspace_id", body=mock.ANY
        )

        assert (
            mock_client.cloud_space_service_create_lightning_run.call_args.kwargs["body"]
            .works[0]
            .spec.data_connection_mounts
            == cloudspace.code_config.data_connection_mounts
        )

        mock_client.cloud_space_service_create_lightning_run_instance.assert_called_once_with(
            project_id="project_id", cloudspace_id="cloudspace_id", id="run_id", body=mock.ANY
        )

        assert mock_client.cloud_space_service_create_lightning_run_instance.call_args.kwargs["body"].name == "run_name"


@mock.patch("lightning.app.core.queues.QueuingSystem", MagicMock())
@mock.patch("lightning.app.runners.backends.cloud.LightningClient", MagicMock())
def test_get_project(monkeypatch):
    mock_client = mock.MagicMock()
    monkeypatch.setattr(cloud, "CloudBackend", mock.MagicMock(return_value=mock_client))
    app = mock.MagicMock(spec=LightningApp)
    cloud.CloudRuntime(app=app, entrypoint=Path("entrypoint.py"))

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
    ret = _get_project(mock_client)
    assert ret.project_id == "test-project-id1"


def write_file_of_size(path, size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.seek(size)
        f.write(b"\0")


@mock.patch("lightning.app.core.queues.QueuingSystem", MagicMock())
@mock.patch("lightning.app.runners.backends.cloud.LightningClient", MagicMock())
def test_check_uploaded_folder(monkeypatch, tmpdir, caplog):
    app = MagicMock()
    root = Path(tmpdir)
    repo = LocalSourceCodeDir(root)
    backend = cloud.CloudRuntime(app)
    with caplog.at_level(logging.WARN):
        backend._validate_repo(root, repo)
    assert caplog.messages == []

    # write some files to assert the message below.
    write_file_of_size(root / "a.png", 4 * 1000 * 1000)
    write_file_of_size(root / "b.txt", 5 * 1000 * 1000)
    write_file_of_size(root / "c.jpg", 6 * 1000 * 1000)

    repo._non_ignored_files = None  # force reset
    with caplog.at_level(logging.WARN):
        backend._validate_repo(root, repo)
    assert f"Your application folder '{root.absolute()}' is more than 2 MB" in caplog.text
    assert "The total size is 15.0 MB" in caplog.text
    assert "4 files were uploaded" in caplog.text
    assert "files:\n6.0 MB: c.jpg\n5.0 MB: b.txt\n4.0 MB: a.png\nPerhaps" in caplog.text  # tests the order
    assert "adding them to `.lightningignore`." in caplog.text
    assert "lightningingore` attribute in a Flow or Work" in caplog.text


@mock.patch("lightning.app.core.queues.QueuingSystem", MagicMock())
@mock.patch("lightning.app.runners.backends.cloud.LightningClient", MagicMock())
def test_project_has_sufficient_credits():
    app = mock.MagicMock(spec=LightningApp)
    cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=Path("entrypoint.py"))
    credits_and_test_value = [
        [0.3, True],
        [1, False],
        [1.1, False],
    ]
    for balance, result in credits_and_test_value:
        project = V1Membership(name="test-project1", project_id="test-project-id1", balance=balance)
        assert cloud_runtime._resolve_needs_credits(project) is result


@pytest.mark.parametrize(
    "lines",
    [
        [
            "import this_package_is_not_real",
            "from lightning.app import LightningApp",
            "from lightning.app.testing.helpers import EmptyWork",
            "app = LightningApp(EmptyWork())",
        ],
        [
            "from this_package_is_not_real import this_module_is_not_real",
            "from lightning.app import LightningApp",
            "from lightning.app.testing.helpers import EmptyWork",
            "app = LightningApp(EmptyWork())",
        ],
        [
            "import this_package_is_not_real",
            "from this_package_is_not_real import this_module_is_not_real",
            "from lightning.app import LightningApp",
            "from lightning.app.testing.helpers import EmptyWork",
            "app = LightningApp(EmptyWork())",
        ],
        [
            "import this_package_is_not_real",
            "from lightning.app import LightningApp",
            "from lightning.app.core.flow import _RootFlow",
            "from lightning.app.testing.helpers import EmptyWork",
            "class MyFlow(_RootFlow):",
            "    def configure_layout(self):",
            "        return [{'name': 'test', 'content': this_package_is_not_real()}]",
            "app = LightningApp(MyFlow(EmptyWork()))",
        ],
    ],
)
@pytest.mark.skipif(sys.platform != "linux", reason="Causing conflicts on non-linux")
def test_load_app_from_file_mock_imports(tmpdir, lines):
    path = copy(sys.path)
    app_file = os.path.join(tmpdir, "app.py")

    with open(app_file, "w") as f:
        f.write("\n".join(lines))

    app = CloudRuntime.load_app_from_file(app_file)
    assert isinstance(app, LightningApp)
    assert isinstance(app.root.work, EmptyWork)

    # Cleanup PATH to prevent conflict with other tests
    sys.path = path
    os.remove(app_file)


def test_load_app_from_file():
    test_script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "scripts")

    app = CloudRuntime.load_app_from_file(
        os.path.join(test_script_dir, "app_with_env.py"),
    )
    assert app.works[0].cloud_compute.name == "cpu-small"

    app = CloudRuntime.load_app_from_file(
        os.path.join(test_script_dir, "app_with_env.py"),
        env_vars={"COMPUTE_NAME": "foo"},
    )
    assert app.works[0].cloud_compute.name == "foo"


@pytest.mark.parametrize(
    ("print_format", "expected"),
    [
        (
            "web",
            [
                {
                    "displayName": "",
                    "name": "root.work",
                    "spec": {
                        "buildSpec": {
                            "commands": [],
                            "pythonDependencies": {"packageManager": "PACKAGE_MANAGER_PIP", "packages": ""},
                        },
                        "dataConnectionMounts": [],
                        "drives": [],
                        "networkConfig": [{"name": "*", "port": "*"}],
                        "userRequestedComputeConfig": {
                            "count": 1,
                            "diskSize": 0,
                            "name": "cpu-small",
                            "preemptible": "*",
                            "shmSize": 0,
                        },
                    },
                }
            ],
        ),
        (
            "gallery",
            [
                {
                    "display_name": "",
                    "name": "root.work",
                    "spec": {
                        "build_spec": {
                            "commands": [],
                            "python_dependencies": {"package_manager": "PACKAGE_MANAGER_PIP", "packages": ""},
                        },
                        "data_connection_mounts": [],
                        "drives": [],
                        "network_config": [{"name": "*", "port": "*"}],
                        "user_requested_compute_config": {
                            "count": 1,
                            "disk_size": 0,
                            "name": "cpu-small",
                            "preemptible": "*",
                            "shm_size": 0,
                        },
                    },
                }
            ],
        ),
    ],
)
def test_print_specs(tmpdir, caplog, monkeypatch, print_format, expected):
    entrypoint = Path(tmpdir) / "entrypoint.py"
    entrypoint.touch()

    mock_client = mock.MagicMock()
    mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(name="test-project", project_id="test-project-id")]
    )
    mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
        V1ListLightningappInstancesResponse(lightningapps=[])
    )
    cloud_backend = mock.MagicMock(client=mock_client)
    monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))

    cloud_runtime = cloud.CloudRuntime(app=LightningApp(EmptyWork()), entrypoint=entrypoint)

    cloud.LIGHTNING_CLOUD_PRINT_SPECS = print_format

    try:
        with caplog.at_level(logging.INFO), contextlib.suppress(SystemExit):
            cloud_runtime.dispatch()

        lines = caplog.text.split("\n")

        expected = re.escape(str(expected).replace("'", '"').replace(" ", "")).replace('"\\*"', "(.*)")
        expected = "INFO(.*)works: " + expected
        assert any(re.fullmatch(expected, line) for line in lines)
    finally:
        cloud.LIGHTNING_CLOUD_PRINT_SPECS = None


def test_incompatible_cloud_compute_and_build_config(monkeypatch):
    """Test that an exception is raised when a build config has a custom image defined, but the cloud compute is the
    default.

    This combination is not supported by the platform.

    """
    mock_client = mock.MagicMock()
    cloud_backend = mock.MagicMock(client=mock_client)
    monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))

    class Work(LightningWork):
        def __init__(self):
            super().__init__()
            self.cloud_compute = CloudCompute(name="default")
            # TODO: Remove me
            self.cloud_compute.name = "default"
            self.cloud_build_config = BuildConfig(image="custom")

        def run(self):
            pass

    app = MagicMock()
    app.works = [Work()]

    with pytest.raises(ValueError, match="You requested a custom base image for the Work with name"):
        CloudRuntime(app=app)._validate_work_build_specs_and_compute()


def test_programmatic_lightningignore(monkeypatch, caplog, tmpdir):
    path = Path(tmpdir)
    entrypoint = path / "entrypoint.py"
    entrypoint.touch()

    mock_client = mock.MagicMock()
    mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(name="test-project", project_id="test-project-id")]
    )
    mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
        V1ListLightningappInstancesResponse(lightningapps=[])
    )
    mock_client.cloud_space_service_create_lightning_run.return_value = V1LightningRun(cluster_id="test")
    cloud_backend = mock.MagicMock(client=mock_client)
    monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))

    class MyWork(LightningWork):
        def __init__(self):
            super().__init__()
            self.lightningignore += ("foo", "lightning_logs")

        def run(self):
            with pytest.raises(RuntimeError, match="w.lightningignore` does not"):
                self.lightningignore += ("foobar",)

    class MyFlow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.lightningignore = ("foo",)
            self.w = MyWork()

        def run(self):
            with pytest.raises(RuntimeError, match="root.lightningignore` does not"):
                self.lightningignore = ("baz",)
            self.w.run()

    flow = MyFlow()
    app = LightningApp(flow)

    monkeypatch.setattr(app, "_update_index_file", mock.MagicMock())

    cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=entrypoint)
    monkeypatch.setattr(LocalSourceCodeDir, "upload", mock.MagicMock())

    # write some files
    write_file_of_size(path / "a.txt", 5 * 1000 * 1000)
    write_file_of_size(path / "foo.png", 4 * 1000 * 1000)
    write_file_of_size(path / "lightning_logs" / "foo.ckpt", 6 * 1000 * 1000)
    # also an actual .lightningignore file
    (path / ".lightningignore").write_text("foo.png")

    with mock.patch(
        "lightning.app.runners.cloud._parse_lightningignore", wraps=_parse_lightningignore
    ) as parse_mock, mock.patch(
        "lightning.app.source_code.local._copytree", wraps=_copytree
    ) as copy_mock, caplog.at_level(logging.WARN):
        cloud_runtime.dispatch()

    parse_mock.assert_called_once_with(("foo", "foo", "lightning_logs"))
    assert copy_mock.mock_calls[0].kwargs["ignore_functions"][0].args[1] == {"lightning_logs", "foo"}

    assert f"Your application folder '{path.absolute()}' is more than 2 MB" in caplog.text
    assert "The total size is 5.0 MB" in caplog.text
    assert "2 files were uploaded"  # a.txt and .lightningignore
    assert "files:\n5.0 MB: a.txt\nPerhaps" in caplog.text  # only this file appears

    flow.run()


def test_default_lightningignore(monkeypatch, caplog, tmpdir):
    path = Path(tmpdir)
    entrypoint = path / "entrypoint.py"
    entrypoint.touch()

    mock_client = mock.MagicMock()
    mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(name="test-project", project_id="test-project-id")]
    )
    mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
        V1ListLightningappInstancesResponse(lightningapps=[])
    )
    mock_client.cloud_space_service_create_lightning_run.return_value = V1LightningRun(cluster_id="test")
    cloud_backend = mock.MagicMock(client=mock_client)
    monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))

    class MyWork(LightningWork):
        def run(self):
            pass

    app = LightningApp(MyWork())

    cloud_runtime = cloud.CloudRuntime(app=app, entrypoint=entrypoint)
    monkeypatch.setattr(LocalSourceCodeDir, "upload", mock.MagicMock())

    # write some files
    write_file_of_size(path / "a.txt", 5 * 1000 * 1000)
    write_file_of_size(path / "venv" / "foo.txt", 4 * 1000 * 1000)

    assert not (path / ".lightningignore").exists()

    with mock.patch(
        "lightning.app.runners.cloud._parse_lightningignore", wraps=_parse_lightningignore
    ) as parse_mock, mock.patch(
        "lightning.app.source_code.local._copytree", wraps=_copytree
    ) as copy_mock, caplog.at_level(logging.WARN):
        cloud_runtime.dispatch()

    parse_mock.assert_called_once_with(())
    assert copy_mock.mock_calls[0].kwargs["ignore_functions"][0].args[1] == set()

    assert (path / ".lightningignore").exists()

    assert f"Your application folder '{path.absolute()}' is more than 2 MB" in caplog.text
    assert "The total size is 5.0 MB" in caplog.text
    assert "2 files were uploaded"  # a.txt and .lightningignore
    assert "files:\n5.0 MB: a.txt\nPerhaps" in caplog.text  # only this file appears


@pytest.mark.parametrize(
    ("project", "run_instance", "user", "tab", "lightning_cloud_url", "expected_url"),
    [
        # Old style
        (
            V1Membership(),
            Externalv1LightningappInstance(id="test-app-id"),
            V1GetUserResponse(username="tester", features=V1UserFeatures()),
            "logs",
            "https://lightning.ai",
            "https://lightning.ai/tester/apps/test-app-id/logs",
        ),
        (
            V1Membership(),
            Externalv1LightningappInstance(id="test-app-id"),
            V1GetUserResponse(username="tester", features=V1UserFeatures()),
            "logs",
            "http://localhost:9800",
            "http://localhost:9800/tester/apps/test-app-id/logs",
        ),
        # New style
        (
            V1Membership(name="tester's project"),
            Externalv1LightningappInstance(name="test/job"),
            V1GetUserResponse(username="tester", features=V1UserFeatures(project_selector=True)),
            "logs",
            "https://lightning.ai",
            "https://lightning.ai/tester/tester%27s%20project/jobs/test%2Fjob/logs",
        ),
        (
            V1Membership(name="tester's project"),
            Externalv1LightningappInstance(name="test/job"),
            V1GetUserResponse(username="tester", features=V1UserFeatures(project_selector=True)),
            "logs",
            "https://localhost:9800",
            "https://localhost:9800/tester/tester%27s%20project/jobs/test%2Fjob/logs",
        ),
    ],
)
def test_get_app_url(monkeypatch, project, run_instance, user, tab, lightning_cloud_url, expected_url):
    mock_client = mock.MagicMock()
    mock_client.auth_service_get_user.return_value = user
    cloud_backend = mock.MagicMock(client=mock_client)
    monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))

    runtime = CloudRuntime()

    with mock.patch(
        "lightning.app.runners.cloud.get_lightning_cloud_url", mock.MagicMock(return_value=lightning_cloud_url)
    ):
        assert runtime._get_app_url(project, run_instance, tab) == expected_url


@pytest.mark.parametrize(
    ("user", "project", "cloudspace_name", "tab", "lightning_cloud_url", "expected_url"),
    [
        (
            V1GetUserResponse(username="tester", features=V1UserFeatures()),
            V1Membership(name="default-project"),
            "test/cloudspace",
            "code",
            "https://lightning.ai",
            "https://lightning.ai/tester/default-project/apps/test%2Fcloudspace/code",
        ),
        (
            V1GetUserResponse(username="tester", features=V1UserFeatures()),
            V1Membership(name="Awesome Project"),
            "The Best CloudSpace ever",
            "web-ui",
            "http://localhost:9800",
            "http://localhost:9800/tester/Awesome%20Project/apps/The%20Best%20CloudSpace%20ever/web-ui",
        ),
    ],
)
def test_get_cloudspace_url(monkeypatch, user, project, cloudspace_name, tab, lightning_cloud_url, expected_url):
    mock_client = mock.MagicMock()
    mock_client.auth_service_get_user.return_value = user
    cloud_backend = mock.MagicMock(client=mock_client)
    monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))

    runtime = CloudRuntime()

    with mock.patch(
        "lightning.app.runners.cloud.get_lightning_cloud_url", mock.MagicMock(return_value=lightning_cloud_url)
    ):
        assert runtime._get_cloudspace_url(project, cloudspace_name, tab) == expected_url
