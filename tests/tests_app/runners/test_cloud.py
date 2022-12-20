import logging
import os
import re
import sys
from copy import copy
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
from lightning_cloud.openapi import (
    Body8,
    Body9,
    Externalv1Cluster,
    Externalv1LightningappInstance,
    Gridv1ImageSpec,
    IdGetBody,
    V1BuildSpec,
    V1ClusterSpec,
    V1ClusterType,
    V1DependencyFileInfo,
    V1Drive,
    V1DriveSpec,
    V1DriveStatus,
    V1DriveType,
    V1EnvVar,
    V1GetClusterResponse,
    V1LightningappInstanceState,
    V1LightningappRelease,
    V1LightningworkDrives,
    V1LightningworkSpec,
    V1ListClustersResponse,
    V1ListLightningappInstancesResponse,
    V1ListLightningappsV2Response,
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
    V1UserRequestedComputeConfig,
    V1UserRequestedFlowComputeConfig,
    V1Work,
)

from lightning_app import BuildConfig, LightningApp, LightningFlow, LightningWork
from lightning_app.runners import backends, cloud, CloudRuntime
from lightning_app.runners.cloud import (
    _generate_works_json_gallery,
    _generate_works_json_web,
    _validate_build_spec_and_compute,
)
from lightning_app.source_code.copytree import _copytree, _parse_lightningignore
from lightning_app.source_code.local import LocalSourceCodeDir
from lightning_app.storage import Drive, Mount
from lightning_app.testing.helpers import EmptyWork
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.dependency_caching import get_hash
from lightning_app.utilities.packaging.cloud_compute import CloudCompute


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


def get_cloud_runtime_request_body(**kwargs) -> "Body8":
    default_request_body = dict(
        app_entrypoint_file=mock.ANY,
        enable_app_server=True,
        is_headless=True,
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


@pytest.fixture
def cloud_backend(monkeypatch):
    cloud_backend = mock.MagicMock()
    monkeypatch.setattr(cloud, "LocalSourceCodeDir", mock.MagicMock())
    monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", mock.MagicMock())
    monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))
    return cloud_backend


@pytest.fixture
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

        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse(
            [
                Externalv1Cluster(id=DEFAULT_CLUSTER),
            ]
        )
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

        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint_file="entrypoint.py")
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        with pytest.raises(ValueError, match="that cluster doesn't exist"):
            cloud_runtime.dispatch(name=app_name, cluster_id="unknown-cluster")

    @pytest.mark.parametrize(
        "old_cluster,new_cluster",
        [
            ("test", "other"),
            ("test", "test"),
            (None, None),
            (None, "litng-ai-03"),
            ("litng-ai-03", None),
        ],
    )
    def test_new_instance_on_different_cluster(self, cloud_backend, project_id, old_cluster, new_cluster):
        app_name = "test-app"

        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="Default Project", project_id=project_id)]
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease(
            cluster_id=new_cluster
        )

        # Note:
        # backend converts "None" cluster to "litng-ai-03"
        # dispatch should receive None, but API calls should return "litng-ai-03"
        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse(
            [
                Externalv1Cluster(id=old_cluster or DEFAULT_CLUSTER),
                Externalv1Cluster(id=new_cluster or DEFAULT_CLUSTER),
            ]
        )

        mock_client.projects_service_list_project_cluster_bindings.return_value = V1ListProjectClusterBindingsResponse(
            clusters=[
                V1ProjectClusterBinding(cluster_id=old_cluster or DEFAULT_CLUSTER),
                V1ProjectClusterBinding(cluster_id=new_cluster or DEFAULT_CLUSTER),
            ]
        )

        # Mock all clusters as global clusters
        mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: V1GetClusterResponse(
            id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
        )

        cloud_backend.client = mock_client

        app = mock.MagicMock()
        app.flows = []
        app.frontend = {}

        existing_app = MagicMock()
        existing_app.name = app_name
        existing_app.id = "test-id"
        mock_client.lightningapp_v2_service_list_lightningapps_v2.return_value = V1ListLightningappsV2Response(
            lightningapps=[existing_app]
        )

        existing_instance = MagicMock()
        existing_instance.name = app_name
        existing_instance.status.phase = V1LightningappInstanceState.STOPPED
        existing_instance.spec.cluster_id = old_cluster or DEFAULT_CLUSTER
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[existing_instance])
        )

        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint_file="entrypoint.py")
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        # This is the main assertion:
        # we have an existing instance on `cluster-001`
        # but we want to run this app on `cluster-002`
        cloud_runtime.dispatch(name=app_name, cluster_id=new_cluster)

        if new_cluster != old_cluster and None not in (old_cluster, new_cluster):
            # If we switched cluster, check that a new name was used which starts with the old name
            mock_client.lightningapp_v2_service_create_lightningapp_release_instance.assert_called_once()
            args = mock_client.lightningapp_v2_service_create_lightningapp_release_instance.call_args
            assert args[1]["body"].name != app_name
            assert args[1]["body"].name.startswith(app_name)
            assert args[1]["body"].cluster_id == new_cluster

    def test_running_deleted_app(self, cloud_backend, project_id):
        """Deleted apps show up in list apps but not in list instances.

        This tests that we don't try to reacreate a previously deleted app.
        """
        app_name = "test-app"

        mock_client = mock.MagicMock()
        mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
            memberships=[V1Membership(name="Default Project", project_id=project_id)]
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease(
            cluster_id=DEFAULT_CLUSTER
        )

        mock_client.cluster_service_list_clusters.return_value = V1ListClustersResponse(
            [
                Externalv1Cluster(id=DEFAULT_CLUSTER),
            ]
        )

        mock_client.projects_service_list_project_cluster_bindings.return_value = V1ListProjectClusterBindingsResponse(
            clusters=[
                V1ProjectClusterBinding(cluster_id=DEFAULT_CLUSTER),
            ]
        )

        # Mock all clusters as global clusters
        mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: V1GetClusterResponse(
            id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
        )

        cloud_backend.client = mock_client

        app = mock.MagicMock()
        app.flows = []
        app.frontend = {}

        existing_app = MagicMock()
        existing_app.name = app_name
        existing_app.id = "test-id"
        mock_client.lightningapp_v2_service_list_lightningapps_v2.return_value = V1ListLightningappsV2Response(
            lightningapps=[existing_app]
        )

        # Simulate the app as deleted so no instance to return
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=[])
        )

        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint_file="entrypoint.py")
        cloud_runtime._check_uploaded_folder = mock.MagicMock()

        cloud_runtime.dispatch(name=app_name)

        # Check that a new name was used which starts with and does not equal the old name
        mock_client.lightningapp_v2_service_create_lightningapp_release_instance.assert_called_once()
        args = mock_client.lightningapp_v2_service_create_lightningapp_release_instance.call_args
        assert args[1]["body"].name != app_name
        assert args[1]["body"].name.startswith(app_name)

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
        app.is_headless = False
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
            is_headless=False,
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
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease()
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
            is_headless=False,
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
    @pytest.mark.parametrize(
        "lightningapps,start_with_flow",
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
        mock_client.lightningapp_v2_service_list_lightningapps_v2.return_value = V1ListLightningappsV2Response(
            lightningapps=lightningapps
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=lightningapps)
        )
        mock_client.projects_service_list_project_cluster_bindings.return_value = V1ListProjectClusterBindingsResponse(
            clusters=[
                V1ProjectClusterBinding(cluster_id="test"),
            ]
        )
        mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: V1GetClusterResponse(
            id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease()
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
        app.is_headless = False

        work = MyWork(start_with_flow=start_with_flow, cloud_compute=CloudCompute("custom"))
        work._name = "test-work"
        work._cloud_build_config.build_commands = lambda: ["echo 'start'"]
        work._cloud_build_config.requirements = ["torch==1.0.0", "numpy==1.0.0"]
        work._cloud_build_config.image = "random_base_public_image"
        work._cloud_compute.disk_size = 0
        work._port = 8080

        app.works = [work]
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
                is_headless=False,
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
                        ),
                    )
                ]
            else:
                expected_body.works = []

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
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease()
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
        Path(source_code_root_dir / ".lightning").write_text("name: myapp")
        requirements_file = Path(source_code_root_dir / "requirements.txt")
        Path(requirements_file).touch()
        (source_code_root_dir / "entrypoint.py").touch()

        mock_client = mock.MagicMock()
        if lightningapps:
            lightningapps[0].name = "myapp"
            lightningapps[0].status.phase = V1LightningappInstanceState.STOPPED
            lightningapps[0].spec.cluster_id = "test"
        mock_client.lightningapp_v2_service_list_lightningapps_v2.return_value = V1ListLightningappsV2Response(
            lightningapps=lightningapps
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=lightningapps)
        )
        mock_client.projects_service_list_project_cluster_bindings.return_value = V1ListProjectClusterBindingsResponse(
            clusters=[
                V1ProjectClusterBinding(cluster_id="test"),
            ]
        )
        mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: V1GetClusterResponse(
            id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease()
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
                is_headless=False,
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
    @mock.patch("lightning_app.core.constants.ENABLE_APP_COMMENT_COMMAND_EXECUTION", True)
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
                V1ListProjectClusterBindingsResponse(
                    clusters=[
                        V1ProjectClusterBinding(cluster_id="test"),
                    ]
                )
            )
            mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: V1GetClusterResponse(
                id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
            )
        mock_client.lightningapp_v2_service_list_lightningapps_v2.return_value = V1ListLightningappsV2Response(
            lightningapps=lightningapps
        )
        mock_client.lightningapp_instance_service_list_lightningapp_instances.return_value = (
            V1ListLightningappInstancesResponse(lightningapps=lightningapps)
        )
        mock_client.lightningapp_v2_service_create_lightningapp_release.return_value = V1LightningappRelease()
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
        cloud_runtime = cloud.CloudRuntime(app=app, entrypoint_file=(source_code_root_dir / "entrypoint.py"))
        monkeypatch.setattr(
            "lightning_app.runners.cloud._get_project",
            lambda x: V1Membership(name="test-project", project_id="test-project-id"),
        )
        cloud_runtime.run_app_comment_commands = True
        cloud_runtime.dispatch()

        if lightningapps:
            expected_body = Body8(
                description=None,
                local_source=True,
                app_entrypoint_file="entrypoint.py",
                enable_app_server=True,
                is_headless=False,
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
                                name="custom", count=1, disk_size=0, shm_size=0, preemptible=mock.ANY
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                            cluster_id=mock.ANY,
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
                project_id="test-project-id",
                app_id=mock.ANY,
                id=mock.ANY,
                body=Body9(
                    desired_state=V1LightningappInstanceState.STOPPED,
                    name=mock.ANY,
                    env=[V1EnvVar(name="ENABLE_APP_COMMENT_COMMAND_EXECUTION", value="1")],
                    queue_server_type=mock.ANY,
                ),
            )

    @mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
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
            mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: V1GetClusterResponse(
                id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
            )
        mock_client.lightningapp_v2_service_list_lightningapps_v2.return_value = V1ListLightningappsV2Response(
            lightningapps=lightningapps
        )
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

            expected_body_option_1 = Body8(
                description=None,
                local_source=True,
                app_entrypoint_file="entrypoint.py",
                enable_app_server=True,
                is_headless=False,
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
                                name="custom",
                                count=1,
                                disk_size=0,
                                shm_size=0,
                                preemptible=False,
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
                        ),
                    )
                ],
            )

            expected_body_option_2 = Body8(
                description=None,
                local_source=True,
                app_entrypoint_file="entrypoint.py",
                enable_app_server=True,
                is_headless=False,
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
                                name="custom",
                                count=1,
                                disk_size=0,
                                shm_size=0,
                                preemptible=False,
                            ),
                            network_config=[V1NetworkConfig(name=mock.ANY, host=None, port=8080)],
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
            mock_client.cluster_service_get_cluster.side_effect = lambda cluster_id: V1GetClusterResponse(
                id=cluster_id, spec=V1ClusterSpec(cluster_type=V1ClusterType.GLOBAL)
            )
        mock_client.lightningapp_v2_service_list_lightningapps_v2.return_value = V1ListLightningappsV2Response(
            lightningapps=lightningapps
        )
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
                is_headless=False,
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


def write_file_of_size(path, size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.seek(size)
        f.write(b"\0")


@mock.patch("lightning_app.core.queues.QueuingSystem", MagicMock())
@mock.patch("lightning_app.runners.backends.cloud.LightningClient", MagicMock())
def test_check_uploaded_folder(monkeypatch, tmpdir, caplog):
    app = MagicMock()
    root = Path(tmpdir)
    repo = LocalSourceCodeDir(root)
    backend = cloud.CloudRuntime(app)
    with caplog.at_level(logging.WARN):
        backend._check_uploaded_folder(root, repo)
    assert caplog.messages == []

    # write some files to assert the message below.
    write_file_of_size(root / "a.png", 4 * 1000 * 1000)
    write_file_of_size(root / "b.txt", 5 * 1000 * 1000)
    write_file_of_size(root / "c.jpg", 6 * 1000 * 1000)

    repo._non_ignored_files = None  # force reset
    with caplog.at_level(logging.WARN):
        backend._check_uploaded_folder(root, repo)
    assert f"Your application folder '{root.absolute()}' is more than 2 MB" in caplog.text
    assert "The total size is 15.0 MB" in caplog.text
    assert "3 files were uploaded" in caplog.text
    assert "files:\n6.0 MB: c.jpg\n5.0 MB: b.txt\n4.0 MB: a.png\nPerhaps" in caplog.text  # tests the order
    assert "adding them to `.lightningignore`." in caplog.text
    assert "lightningingore` attribute in a Flow or Work" in caplog.text


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


@pytest.mark.parametrize(
    "lines",
    [
        [
            "import this_package_is_not_real",
            "from lightning_app import LightningApp",
            "from lightning_app.testing.helpers import EmptyWork",
            "app = LightningApp(EmptyWork())",
        ],
        [
            "from this_package_is_not_real import this_module_is_not_real",
            "from lightning_app import LightningApp",
            "from lightning_app.testing.helpers import EmptyWork",
            "app = LightningApp(EmptyWork())",
        ],
        [
            "import this_package_is_not_real",
            "from this_package_is_not_real import this_module_is_not_real",
            "from lightning_app import LightningApp",
            "from lightning_app.testing.helpers import EmptyWork",
            "app = LightningApp(EmptyWork())",
        ],
        [
            "import this_package_is_not_real",
            "from lightning_app import LightningApp",
            "from lightning_app.core.flow import _RootFlow",
            "from lightning_app.testing.helpers import EmptyWork",
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


@pytest.mark.parametrize(
    "generator,expected",
    [
        (
            _generate_works_json_web,
            [
                {
                    "name": "root.work",
                    "spec": {
                        "buildSpec": {
                            "commands": [],
                            "pythonDependencies": {"packageManager": "PACKAGE_MANAGER_PIP", "packages": ""},
                        },
                        "drives": [],
                        "networkConfig": [{"name": "*", "port": "*"}],
                        "userRequestedComputeConfig": {
                            "count": 1,
                            "diskSize": 0,
                            "name": "default",
                            "preemptible": "*",
                            "shmSize": 0,
                        },
                    },
                }
            ],
        ),
        (
            _generate_works_json_gallery,
            [
                {
                    "name": "root.work",
                    "spec": {
                        "build_spec": {
                            "commands": [],
                            "python_dependencies": {"package_manager": "PACKAGE_MANAGER_PIP", "packages": ""},
                        },
                        "drives": [],
                        "network_config": [{"name": "*", "port": "*"}],
                        "user_requested_compute_config": {
                            "count": 1,
                            "disk_size": 0,
                            "name": "default",
                            "preemptible": "*",
                            "shm_size": 0,
                        },
                    },
                }
            ],
        ),
    ],
)
@pytest.mark.skipif(sys.platform != "linux", reason="Causing conflicts on non-linux")
def test_generate_works_json(tmpdir, generator, expected):
    path = copy(sys.path)
    app_file = os.path.join(tmpdir, "app.py")

    with open(app_file, "w") as f:
        lines = [
            "from lightning_app import LightningApp",
            "from lightning_app.testing.helpers import EmptyWork",
            "app = LightningApp(EmptyWork())",
        ]
        f.write("\n".join(lines))

    works_string = generator(app_file)
    expected = re.escape(str(expected).replace("'", '"').replace(" ", "")).replace('"\\*"', "(.*)")
    assert re.fullmatch(expected, works_string)

    # Cleanup PATH to prevent conflict with other tests
    sys.path = path
    os.remove(app_file)


def test_incompatible_cloud_compute_and_build_config():
    """Test that an exception is raised when a build config has a custom image defined, but the cloud compute is
    the default.

    This combination is not supported by the platform.
    """

    class Work(LightningWork):
        def __init__(self):
            super().__init__()
            self.cloud_compute = CloudCompute(name="default")
            self.cloud_build_config = BuildConfig(image="custom")

        def run(self):
            pass

    with pytest.raises(ValueError, match="You requested a custom base image for the Work with name"):
        _validate_build_spec_and_compute(Work())


def test_programmatic_lightningignore(monkeypatch, caplog, tmpdir):
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

    path = Path(tmpdir)
    cloud_runtime = cloud.CloudRuntime(app=app, entrypoint_file=path / "entrypoint.py")
    monkeypatch.setattr(LocalSourceCodeDir, "upload", mock.MagicMock())

    # write some files
    write_file_of_size(path / "a.txt", 5 * 1000 * 1000)
    write_file_of_size(path / "foo.png", 4 * 1000 * 1000)
    write_file_of_size(path / "lightning_logs" / "foo.ckpt", 6 * 1000 * 1000)
    # also an actual .lightningignore file
    (path / ".lightningignore").write_text("foo.png")

    with mock.patch(
        "lightning_app.runners.cloud._parse_lightningignore", wraps=_parse_lightningignore
    ) as parse_mock, mock.patch(
        "lightning_app.source_code.local._copytree", wraps=_copytree
    ) as copy_mock, caplog.at_level(
        logging.WARN
    ):
        cloud_runtime.dispatch()

    parse_mock.assert_called_once_with(("foo", "foo", "lightning_logs"))
    assert copy_mock.mock_calls[0].kwargs["ignore_functions"][0].args[1] == {"lightning_logs", "foo"}

    assert f"Your application folder '{path.absolute()}' is more than 2 MB" in caplog.text
    assert "The total size is 5.0 MB" in caplog.text
    assert "2 files were uploaded"  # a.txt and .lightningignore
    assert "files:\n5.0 MB: a.txt\nPerhaps" in caplog.text  # only this file appears

    flow.run()


def test_default_lightningignore(monkeypatch, caplog, tmpdir):
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
    cloud_backend = mock.MagicMock(client=mock_client)
    monkeypatch.setattr(backends, "CloudBackend", mock.MagicMock(return_value=cloud_backend))

    class MyWork(LightningWork):
        def run(self):
            pass

    app = LightningApp(MyWork())

    path = Path(tmpdir)
    cloud_runtime = cloud.CloudRuntime(app=app, entrypoint_file=path / "entrypoint.py")
    monkeypatch.setattr(LocalSourceCodeDir, "upload", mock.MagicMock())

    # write some files
    write_file_of_size(path / "a.txt", 5 * 1000 * 1000)
    write_file_of_size(path / "venv" / "foo.txt", 4 * 1000 * 1000)

    assert not (path / ".lightningignore").exists()

    with mock.patch(
        "lightning_app.runners.cloud._parse_lightningignore", wraps=_parse_lightningignore
    ) as parse_mock, mock.patch(
        "lightning_app.source_code.local._copytree", wraps=_copytree
    ) as copy_mock, caplog.at_level(
        logging.WARN
    ):
        cloud_runtime.dispatch()

    parse_mock.assert_called_once_with(())
    assert copy_mock.mock_calls[0].kwargs["ignore_functions"][0].args[1] == set()

    assert (path / ".lightningignore").exists()

    assert f"Your application folder '{path.absolute()}' is more than 2 MB" in caplog.text
    assert "The total size is 5.0 MB" in caplog.text
    assert "2 files were uploaded"  # a.txt and .lightningignore
    assert "files:\n5.0 MB: a.txt\nPerhaps" in caplog.text  # only this file appears


@pytest.mark.parametrize(
    "lightning_app_instance, lightning_cloud_url, expected_url",
    [
        (
            Externalv1LightningappInstance(id="test-app-id"),
            "https://b975913c4b22eca5f0f9e8eff4c4b1c315340a0d.staging.lightning.ai",
            "https://b975913c4b22eca5f0f9e8eff4c4b1c315340a0d.staging.lightning.ai/me/apps/test-app-id",
        ),
        (
            Externalv1LightningappInstance(id="test-app-id"),
            "http://localhost:9800",
            "http://localhost:9800/me/apps/test-app-id",
        ),
    ],
)
def test_get_app_url(lightning_app_instance, lightning_cloud_url, expected_url):
    with mock.patch(
        "lightning_app.runners.cloud.get_lightning_cloud_url", mock.MagicMock(return_value=lightning_cloud_url)
    ):
        assert CloudRuntime._get_app_url(lightning_app_instance) == expected_url
