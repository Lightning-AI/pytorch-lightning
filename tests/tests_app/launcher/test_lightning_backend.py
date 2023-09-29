import json
import os
from copy import copy
from datetime import datetime
from unittest import mock
from unittest.mock import ANY, MagicMock, Mock

import pytest
from lightning.app import BuildConfig, CloudCompute, LightningWork
from lightning.app.launcher.lightning_backend import CloudBackend
from lightning.app.storage import Drive, Mount
from lightning.app.testing.helpers import EmptyWork
from lightning.app.utilities.enum import WorkFailureReasons, WorkStageStatus
from lightning.app.utilities.exceptions import LightningPlatformException
from lightning_cloud.openapi import Body5, V1DriveType, V1LightningworkState, V1SourceType
from lightning_cloud.openapi.rest import ApiException


class WorkWithDrive(LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drive = None

    def run(self):
        pass


@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_no_update_when_no_works(client_mock):
    cloud_backend = CloudBackend("")
    cloud_backend._get_cloud_work_specs = Mock()
    client_mock.assert_called_once()
    cloud_backend.update_work_statuses(works=[])
    cloud_backend._get_cloud_work_specs.assert_not_called()


@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_no_update_when_all_work_has_started(client_mock):
    cloud_backend = CloudBackend("")
    cloud_backend._get_cloud_work_specs = MagicMock()
    client_mock.assert_called_once()
    started_mock = MagicMock()
    started_mock.has_started = True

    # all works have started
    works = [started_mock, started_mock]
    cloud_backend.update_work_statuses(works=works)
    cloud_backend._get_cloud_work_specs.assert_called_once()


@mock.patch("lightning.app.launcher.lightning_backend.monotonic")
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_no_update_within_interval(client_mock, monotonic_mock):
    cloud_backend = CloudBackend("", status_update_interval=2)
    cloud_backend._get_cloud_work_specs = Mock()
    client_mock.assert_called_once()
    cloud_backend._last_time_updated = 1
    monotonic_mock.return_value = 2

    stopped_mock = Mock()
    stopped_mock.has_started = False

    # not all works have started
    works = [stopped_mock, stopped_mock]

    cloud_backend.update_work_statuses(works=works)
    cloud_backend._get_cloud_work_specs.assert_not_called()


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.launcher.lightning_backend.monotonic")
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_update_within_interval(client_mock, monotonic_mock):
    cloud_backend = CloudBackend("", status_update_interval=2)
    cloud_backend._last_time_updated = 1
    # pretend a lot of time has passed since the last update
    monotonic_mock.return_value = 8

    stopped_mock1 = Mock()
    stopped_mock1.has_started = False
    stopped_mock1.name = "root.mock1"
    stopped_mock2 = Mock()
    stopped_mock2.has_started = False
    stopped_mock2.name = "root.mock2"

    spec1 = Mock()
    spec1.name = "root.mock1"
    spec2 = Mock()
    spec2.name = "root.mock2"

    # not all works have started
    works = [stopped_mock1, stopped_mock2]

    cloud_backend.update_work_statuses(works=works)
    client_mock().lightningwork_service_list_lightningwork.assert_called_with(project_id="project_id", app_id="app_id")

    # TODO: assert calls on the work mocks
    # ...


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_stop_all_works(mock_client):
    work_a = EmptyWork()
    work_a._name = "root.work_a"
    work_a._calls = {
        "latest_call_hash": "some_call_hash",
        "some_call_hash": {
            "statuses": [
                {
                    "stage": WorkStageStatus.FAILED,
                    "timestamp": int(datetime.now().timestamp()),
                    "reason": WorkFailureReasons.USER_EXCEPTION,
                },
            ]
        },
    }

    work_b = EmptyWork()
    work_b._name = "root.work_b"
    work_b._calls = {
        "latest_call_hash": "some_call_hash",
        "some_call_hash": {
            "statuses": [{"stage": WorkStageStatus.RUNNING, "timestamp": int(datetime.now().timestamp()), "reason": ""}]
        },
    }

    cloud_backend = CloudBackend("")

    spec1 = Mock()
    spec1.name = "root.work_a"
    spec1.spec.desired_state = V1LightningworkState.RUNNING
    spec1.status.phase = V1LightningworkState.FAILED
    spec2 = Mock()
    spec2.name = "root.work_b"
    spec2.spec.desired_state = V1LightningworkState.RUNNING

    class BackendMock:
        def __init__(self):
            self.called = 0

        def _get_cloud_work_specs(self, *_):
            value = [spec1, spec2] if not self.called else []
            self.called += 1
            return value

    cloud_backend._get_cloud_work_specs = BackendMock()._get_cloud_work_specs
    cloud_backend.stop_all_works([work_a, work_b])

    mock_client().lightningwork_service_update_lightningwork.assert_called_with(
        project_id="project_id",
        id=ANY,
        spec_lightningapp_instance_id="app_id",
        body=ANY,
    )
    assert spec1.spec.desired_state == V1LightningworkState.RUNNING
    assert spec2.spec.desired_state == V1LightningworkState.STOPPED


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_stop_work(mock_client):
    work = EmptyWork()
    work._name = "root.work"
    work._calls = {
        "latest_call_hash": "some_call_hash",
        "some_call_hash": {
            "statuses": [
                {
                    "stage": WorkStageStatus.RUNNING,
                    "timestamp": int(datetime.now().timestamp()),
                    "reason": "",
                },
            ]
        },
    }

    cloud_backend = CloudBackend("")
    spec1 = Mock()
    spec1.name = "root.work"
    spec1.spec.desired_state = V1LightningworkState.RUNNING

    spec2 = Mock()
    spec2.name = "root.work_b"
    spec2.spec.desired_state = V1LightningworkState.RUNNING

    class BackendMock:
        def __init__(self):
            self.called = 0

        def _get_cloud_work_specs(self, *_):
            value = [spec1, spec2] if not self.called else []
            self.called += 1
            return value

    cloud_backend._get_cloud_work_specs = BackendMock()._get_cloud_work_specs
    cloud_backend.stop_work(MagicMock(), work)

    mock_client().lightningwork_service_update_lightningwork.assert_called_with(
        project_id="project_id",
        id=ANY,
        spec_lightningapp_instance_id="app_id",
        body=ANY,
    )
    assert spec1.spec.desired_state == V1LightningworkState.STOPPED
    assert spec2.spec.desired_state == V1LightningworkState.RUNNING


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_create_work_where_work_does_not_exists(mock_client):
    cloud_backend = CloudBackend("")
    non_matching_spec = Mock()
    app = MagicMock()
    work = EmptyWork(port=1111)
    work._name = "name"

    def lightningwork_service_create_lightningwork(
        project_id: str = None,
        spec_lightningapp_instance_id: str = None,
        body: "Body5" = None,
    ):
        assert project_id == "project_id"
        assert spec_lightningapp_instance_id == "app_id"
        assert len(body.spec.network_config) == 1
        assert body.spec.network_config[0].port == 1111
        assert not body.spec.network_config[0].host
        body.spec.network_config[0].host = "x.lightning.ai"
        return body

    response_mock = Mock()
    response_mock.lightningworks = [non_matching_spec]
    mock_client().lightningwork_service_list_lightningwork.return_value = response_mock
    mock_client().lightningwork_service_create_lightningwork = lightningwork_service_create_lightningwork

    cloud_backend.create_work(app, work)
    assert work._future_url == "https://x.lightning.ai"
    app.work_queues["name"].put.assert_called_once_with(work)

    # testing whether the exception is raised correctly when the backend throws on work creation
    http_resp = MagicMock()
    error_message = "exception generated from test_create_work_where_work_does_not_exists test case"
    http_resp.data = json.dumps({"message": error_message})
    mock_client().lightningwork_service_create_lightningwork = MagicMock()
    mock_client().lightningwork_service_create_lightningwork.side_effect = ApiException(http_resp=http_resp)
    with pytest.raises(LightningPlatformException, match=error_message):
        cloud_backend.create_work(app, work)


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_create_work_with_drives_where_work_does_not_exists(mock_client, tmpdir):
    cloud_backend = CloudBackend("")
    non_matching_spec = Mock()
    app = MagicMock()

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

    work = WorkWithDrive(port=1111)
    work._name = "test-work-name"
    work.drive = mocked_drive

    def lightningwork_service_create_lightningwork(
        project_id: str = None,
        spec_lightningapp_instance_id: str = None,
        body: "Body5" = None,
    ):
        assert project_id == "project_id"
        assert spec_lightningapp_instance_id == "app_id"
        assert len(body.spec.network_config) == 1
        assert body.spec.network_config[0].port == 1111
        assert not body.spec.network_config[0].host
        body.spec.network_config[0].host = "x.lightning.ai"
        assert len(body.spec.drives) == 1
        assert body.spec.drives[0].drive.spec.drive_type == V1DriveType.NO_MOUNT_S3
        assert body.spec.drives[0].drive.spec.source_type == V1SourceType.S3
        assert body.spec.drives[0].drive.spec.source == "lit://foobar"
        assert body.spec.drives[0].drive.metadata.name == "test-work-name.drive"
        for v in body.spec.drives[0].drive.status.to_dict().values():
            assert v is None

        return body

    response_mock = Mock()
    response_mock.lightningworks = [non_matching_spec]
    mock_client().lightningwork_service_list_lightningwork.return_value = response_mock
    mock_client().lightningwork_service_create_lightningwork = lightningwork_service_create_lightningwork

    cloud_backend.create_work(app, work)
    assert work._future_url == "https://x.lightning.ai"
    app.work_queues["test-work-name"].put.assert_called_once_with(work)


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
        "LIGHTNING_PROXY_SCHEME": "http",
    },
)
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_create_work_proxy_http(mock_client, tmpdir):
    cloud_backend = CloudBackend("")
    non_matching_spec = Mock()
    app = MagicMock()

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

    work = WorkWithDrive(port=1111)
    work._name = "test-work-name"
    work.drive = mocked_drive

    def lightningwork_service_create_lightningwork(
        project_id: str = None,
        spec_lightningapp_instance_id: str = None,
        body: "Body5" = None,
    ):
        assert project_id == "project_id"
        assert spec_lightningapp_instance_id == "app_id"
        assert len(body.spec.network_config) == 1
        assert body.spec.network_config[0].port == 1111
        assert not body.spec.network_config[0].host
        body.spec.network_config[0].host = "x.lightning.ai"
        assert len(body.spec.drives) == 1
        assert body.spec.drives[0].drive.spec.drive_type == V1DriveType.NO_MOUNT_S3
        assert body.spec.drives[0].drive.spec.source_type == V1SourceType.S3
        assert body.spec.drives[0].drive.spec.source == "lit://foobar"
        assert body.spec.drives[0].drive.metadata.name == "test-work-name.drive"
        for v in body.spec.drives[0].drive.status.to_dict().values():
            assert v is None

        return body

    response_mock = Mock()
    response_mock.lightningworks = [non_matching_spec]
    mock_client().lightningwork_service_list_lightningwork.return_value = response_mock
    mock_client().lightningwork_service_create_lightningwork = lightningwork_service_create_lightningwork

    cloud_backend.create_work(app, work)
    assert work._future_url == "http://x.lightning.ai"
    app.work_queues["test-work-name"].put.assert_called_once_with(work)


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.utilities.network.create_swagger_client", MagicMock())
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_update_work_with_changed_compute_config_with_mounts(mock_client):
    cloud_backend = CloudBackend("")
    matching_spec = Mock()
    app = MagicMock()
    work = EmptyWork(cloud_compute=CloudCompute("default"), cloud_build_config=BuildConfig(image="image1"))
    work._name = "work_name"

    matching_spec.spec = cloud_backend._work_to_spec(work)
    matching_spec.spec.desired_state = V1LightningworkState.STOPPED
    matching_spec.name = "work_name"

    response_mock = Mock()
    response_mock.lightningworks = [matching_spec]
    mock_client().lightningwork_service_list_lightningwork.return_value = response_mock

    cloud_backend.create_work(app, work)
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs[
            "body"
        ].spec.desired_state
        == V1LightningworkState.RUNNING
    )
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs[
            "body"
        ].spec.user_requested_compute_config.name
        == "cpu-small"
    )
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs[
            "body"
        ].spec.build_spec.image
        == "image1"
    )

    # resetting the values changed in the previous step
    matching_spec.spec.desired_state = V1LightningworkState.STOPPED
    cloud_backend.client.lightningwork_service_update_lightningwork.reset_mock()

    # new work with same name but different compute config
    mount = Mount(source="s3://foo/", mount_path="/foo")
    work = EmptyWork(cloud_compute=CloudCompute("gpu", mounts=mount), cloud_build_config=BuildConfig(image="image2"))
    work._name = "work_name"
    cloud_backend.create_work(app, work)
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs[
            "body"
        ].spec.desired_state
        == V1LightningworkState.RUNNING
    )
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs[
            "body"
        ].spec.user_requested_compute_config.name
        == "gpu"
    )
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs["body"]
        .spec.drives[0]
        .mount_location
        == "/foo"
    )
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs["body"]
        .spec.drives[0]
        .drive.spec.source
        == "s3://foo/"
    )
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs[
            "body"
        ].spec.build_spec.image
        == "image2"
    )


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.utilities.network.create_swagger_client", MagicMock())
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_create_work_where_work_already_exists(mock_client):
    cloud_backend = CloudBackend("")
    matching_spec = Mock()
    app = MagicMock()
    work = EmptyWork(port=1111)
    work._name = "work_name"
    work._backend = cloud_backend

    matching_spec.spec = cloud_backend._work_to_spec(work)
    matching_spec.spec.network_config[0].host = "x.lightning.ai"
    matching_spec.spec.desired_state = V1LightningworkState.STOPPED
    matching_spec.name = "work_name"

    response_mock = Mock()
    response_mock.lightningworks = [matching_spec]
    mock_client().lightningwork_service_list_lightningwork.return_value = response_mock

    cloud_backend.create_work(app, work)
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs[
            "body"
        ].spec.desired_state
        == V1LightningworkState.RUNNING
    )
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs["body"]
        .spec.network_config[0]
        .port
        == 1111
    )
    assert work._future_url == "https://x.lightning.ai"
    app.work_queues["work_name"].put.assert_called_once_with(work)

    # resetting the values changed in the previous step
    matching_spec.spec.desired_state = V1LightningworkState.STOPPED
    cloud_backend.client.lightningwork_service_update_lightningwork.reset_mock()
    app.work_queues["work_name"].put.reset_mock()

    # changing the port
    work._port = 2222
    cloud_backend.create_work(app, work)
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs["body"]
        .spec.network_config[0]
        .port
        == 2222
    )
    app.work_queues["work_name"].put.assert_called_once_with(work)

    # testing whether the exception is raised correctly when the backend throws on work creation
    # resetting the values changed in the previous step
    matching_spec.spec.desired_state = V1LightningworkState.STOPPED
    http_resp = MagicMock()
    error_message = "exception generated from test_create_work_where_work_already_exists test case"
    http_resp.data = json.dumps({"message": error_message})
    mock_client().lightningwork_service_update_lightningwork = MagicMock()
    mock_client().lightningwork_service_update_lightningwork.side_effect = ApiException(http_resp=http_resp)
    with pytest.raises(LightningPlatformException, match=error_message):
        cloud_backend.create_work(app, work)


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.utilities.network.create_swagger_client", MagicMock())
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_create_work_will_have_none_backend(mockclient):
    def queue_put_mock(work):
        # because we remove backend before pushing to queue
        assert work._backend is None

    cloud_backend = CloudBackend("")
    app = MagicMock()
    work = EmptyWork()
    # attaching backend - this will be removed by the queue
    work._backend = cloud_backend
    app.work_queues["work_name"].put = queue_put_mock
    cloud_backend.create_work(app, work)
    # make sure the work still have the backend attached
    assert work._backend == cloud_backend


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.utilities.network.create_swagger_client", MagicMock())
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_update_work_with_changed_compute_config_and_build_spec(mock_client):
    cloud_backend = CloudBackend("")
    matching_spec = Mock()
    app = MagicMock()
    work = EmptyWork(cloud_compute=CloudCompute("default"), cloud_build_config=BuildConfig(image="image1"))
    work._name = "work_name"

    matching_spec.spec = cloud_backend._work_to_spec(work)
    matching_spec.spec.desired_state = V1LightningworkState.STOPPED
    matching_spec.name = "work_name"

    response_mock = Mock()
    response_mock.lightningworks = [matching_spec]
    mock_client().lightningwork_service_list_lightningwork.return_value = response_mock

    cloud_backend.create_work(app, work)
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs[
            "body"
        ].spec.desired_state
        == V1LightningworkState.RUNNING
    )
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs[
            "body"
        ].spec.user_requested_compute_config.name
        == "cpu-small"
    )
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs[
            "body"
        ].spec.build_spec.image
        == "image1"
    )

    # resetting the values changed in the previous step
    matching_spec.spec.desired_state = V1LightningworkState.STOPPED
    cloud_backend.client.lightningwork_service_update_lightningwork.reset_mock()

    # new work with same name but different compute config
    work = EmptyWork(cloud_compute=CloudCompute("gpu"), cloud_build_config=BuildConfig(image="image2"))
    work._name = "work_name"
    cloud_backend.create_work(app, work)
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs[
            "body"
        ].spec.desired_state
        == V1LightningworkState.RUNNING
    )
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs[
            "body"
        ].spec.user_requested_compute_config.name
        == "gpu"
    )
    assert (
        cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args.kwargs[
            "body"
        ].spec.build_spec.image
        == "image2"
    )


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.utilities.network.create_swagger_client", MagicMock())
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_update_work_with_changed_spec_while_work_running(mock_client):
    cloud_backend = CloudBackend("")
    matching_spec = Mock()
    app = MagicMock()
    work = EmptyWork(cloud_compute=CloudCompute("default"), cloud_build_config=BuildConfig(image="image1"))
    work._name = "work_name"

    matching_spec.spec = cloud_backend._work_to_spec(work)
    matching_spec.spec.desired_state = V1LightningworkState.RUNNING
    matching_spec.name = "work_name"

    response_mock = Mock()
    response_mock.lightningworks = [matching_spec]
    mock_client().lightningwork_service_list_lightningwork.return_value = response_mock

    cloud_backend.create_work(app, work)

    # asserting the method is not called
    cloud_backend.client.lightningwork_service_update_lightningwork.assert_not_called()


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.utilities.network.create_swagger_client", MagicMock())
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_update_lightning_app_frontend_new_frontends(mock_client):
    cloud_backend = CloudBackend("")
    cloud_backend.client = mock_client
    mocked_app = MagicMock()
    mocked_app.frontends.keys.return_value = ["frontend2", "frontend1"]
    app_instance_mock = MagicMock()
    app_instance_mock.spec.flow_servers = []
    update_lightning_app_instance_mock = MagicMock()
    mock_client.lightningapp_instance_service_get_lightningapp_instance.return_value = app_instance_mock
    mock_client.lightningapp_instance_service_update_lightningapp_instance.return_value = (
        update_lightning_app_instance_mock
    )
    cloud_backend.update_lightning_app_frontend(mocked_app)
    assert mock_client.lightningapp_instance_service_update_lightningapp_instance.call_count == 1

    # frontends should be sorted
    assert (
        mock_client.lightningapp_instance_service_update_lightningapp_instance.call_args.kwargs["body"]
        .spec.flow_servers[0]
        .name
        == "frontend1"
    )
    assert (
        mock_client.lightningapp_instance_service_update_lightningapp_instance.call_args.kwargs["body"]
        .spec.flow_servers[1]
        .name
        == "frontend2"
    )


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.utilities.network.create_swagger_client", MagicMock())
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_update_lightning_app_frontend_existing_frontends(mock_client):
    cloud_backend = CloudBackend("")
    cloud_backend.client = mock_client
    mocked_app = MagicMock()
    mocked_app.frontends.keys.return_value = ["frontend2", "frontend1"]
    app_instance_mock = MagicMock()
    app_instance_mock.spec.flow_servers = ["frontend2", "frontend1"]
    update_lightning_app_instance_mock = MagicMock()
    mock_client.lightningapp_instance_service_get_lightningapp_instance.return_value = app_instance_mock
    mock_client.lightningapp_instance_service_update_lightningapp_instance.return_value = (
        update_lightning_app_instance_mock
    )
    cloud_backend.update_lightning_app_frontend(mocked_app)

    # the app spec already has the frontends, so no update should be called
    assert mock_client.lightningapp_instance_service_update_lightningapp_instance.call_count == 0
    assert mock_client.lightningapp_instance_service_update_lightningapp_instance.call_count == 0


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.utilities.network.create_swagger_client", MagicMock())
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_stop_app(mock_client):
    cloud_backend = CloudBackend("")
    external_spec = MagicMock()
    mock_client.lightningapp_instance_service_get_lightningapp_instance.return_value = external_spec
    cloud_backend.client = mock_client
    mocked_app = MagicMock()
    cloud_backend.stop_app(mocked_app)
    spec = mock_client.lightningapp_instance_service_update_lightningapp_instance._mock_call_args.kwargs["body"].spec
    assert spec.desired_state == "LIGHTNINGAPP_INSTANCE_STATE_STOPPED"


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_failed_works_during_pending(client_mock):
    cloud_backend = CloudBackend("")
    cloud_work = MagicMock()
    cloud_work.name = "a"
    cloud_work.status.phase = V1LightningworkState.FAILED
    cloud_backend._get_cloud_work_specs = MagicMock(return_value=[cloud_work])

    local_work = MagicMock()
    local_work.status.stage = "pending"
    local_work.name = "a"
    local_work._raise_exception = True

    with pytest.raises(Exception, match="The work a failed during pending phase."):
        # all works have started
        cloud_backend.update_work_statuses(works=[local_work])


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_CLOUD_PROJECT_ID": "project_id",
        "LIGHTNING_CLOUD_APP_ID": "app_id",
    },
)
@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_work_delete(client_mock):
    cloud_backend = CloudBackend("")
    cloud_work = MagicMock()
    cloud_work.name = "a"
    cloud_work.status.phase = V1LightningworkState.RUNNING
    cloud_backend._get_cloud_work_specs = MagicMock(return_value=[cloud_work])

    local_work = MagicMock()
    local_work.status.stage = "running"
    local_work.name = "a"
    local_work._raise_exception = True
    cloud_backend.delete_work(None, local_work)
    call = cloud_backend.client.lightningwork_service_update_lightningwork._mock_call_args_list[0]
    assert call.kwargs["body"].spec.desired_state == V1LightningworkState.DELETED
