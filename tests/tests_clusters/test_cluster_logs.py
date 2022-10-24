import os
import random
import string
from datetime import datetime
from itertools import repeat
from unittest import mock

import pytest
from dateutil.tz import tzutc

from lightning_app.utilities.cluster_logs import (
    _cluster_logs_reader,
    _ClusterLogEvent,
    _ClusterLogEventLabels,
    _parse_log_event,
)
from lightning_app.utilities.exceptions import LogLinesLimitExceeded
from src.lightning_app.testing.testing import run_cli


@pytest.mark.cloud
@pytest.mark.skipif(
    os.environ.get("LIGHTNING_BYOC_CLUSTER_NAME") is None,
    reason="missing LIGHTNING_BYOC_CLUSTER_NAME environment variable",
)
def test_byoc_cluster_logs() -> None:
    # Check a typical retrieving case
    cluster_name = os.environ.get("LIGHTNING_BYOC_CLUSTER_NAME")
    with run_cli(
        [
            "show",
            "cluster",
            "logs",
            cluster_name,
        ]
    ) as (stdout, stderr):
        assert "info" in stdout, f"stdout: {stdout}\nstderr: {stderr}"

    # Check a retrieving case with a small number of lines limit
    cluster_name = os.environ.get("LIGHTNING_BYOC_CLUSTER_NAME")
    with run_cli(
        [
            "show",
            "cluster",
            "logs",
            cluster_name,
            "--limit",
            10,
        ]
    ) as (stdout, stderr):
        assert "info" in stdout, f"stdout: {stdout}\nstderr: {stderr}"

    # Time expanding doesn't break retrieving
    with run_cli(
        [
            "show",
            "cluster",
            "logs",
            cluster_name,
            "--limit",
            10,
            "--from",
            "48 hours ago",
        ]
    ) as (stdout, stderr):
        assert "info" in stdout, f"stdout: {stdout}\nstderr: {stderr}"

    # Time expanding doesn't break retrieving
    with run_cli(
        [
            "show",
            "cluster",
            "logs",
            cluster_name,
            "--limit",
            10,
            "--from",
            "48 hours ago",
        ]
    ) as (stdout, stderr):
        assert "info" in stdout, f"stdout: {stdout}\nstderr: {stderr}"

    # Try non-existing cluster
    letters = string.ascii_letters
    cluster_name = "".join(random.choice(letters) for i in range(10))
    with run_cli(
        [
            "show",
            "cluster",
            "logs",
            cluster_name,
        ]
    ) as (stdout, stderr):
        assert "does not exist" in stdout, f"stdout: {stdout}\nstderr: {stderr}"


@pytest.mark.cloud
@pytest.mark.skipif(
    os.environ.get("LIGHTNING_CLOUD_CLUSTER_NAME") is None,
    reason="missing LIGHTNING_CLOUD_CLUSTER_NAME environment variable",
)
def test_lighting_cloud_logs() -> None:
    # Check a retrieving case from lightning-cloud
    # We shouldn't show lighting-cloud logs, therefore we expect to see an error here
    cluster_name = os.environ.get("LIGHTNING_CLOUD_CLUSTER_NAME" "" "")
    with run_cli(
        [
            "show",
            "cluster",
            "logs",
            cluster_name,
        ]
    ) as (stdout, stderr):
        assert "Error while reading logs" in stdout, f"stdout: {stdout}\nstderr: {stderr}"


def test_cluster_logs_reader():
    logs_api_client = mock.Mock()
    log_socket = mock.Mock()

    def create_cluster_logs_socket(
        cluster_id: str,
        start: float,  # unix timestamp
        end: float,  # unix timestamp
        limit: int,
        on_message_callback,
        on_error_callback,
    ):
        assert start == 0
        assert end == 10
        assert limit == 10

        def run_forever():
            on_message_callback(
                None,
                r"""
            {
              "message": "getting file lock",
              "timestamp": "2022-08-30T00:57:59.370356800Z",
              "labels": {
                "cluster_id": "cluster_id",
                "grid_url": "https://lightning.ai",
                "hostname": "ec2-001",
                "level": "info",
                "logger": "test.logger",
                "path": "/tmp/grid.terraform"
              }
            }
            """,
            )

        log_socket.run_forever = run_forever
        return log_socket

    logs_api_client.create_cluster_logs_socket = mock.MagicMock(
        side_effect=create_cluster_logs_socket,
    )

    logs = list(
        _cluster_logs_reader(
            logs_api_client=logs_api_client,
            cluster_id="cluster_id",
            start=0,
            end=10,
            limit=10,
            follow=False,
        )
    )
    logs_api_client.create_cluster_logs_socket.assert_called_once()

    assert logs == [
        _ClusterLogEvent(
            message="getting file lock",
            timestamp=datetime(2022, 8, 30, 0, 57, 59, 370356, tzinfo=tzutc()),
            labels=_ClusterLogEventLabels(
                cluster_id="cluster_id",
                grid_url="https://lightning.ai",
                hostname="ec2-001",
                level="info",
                logger="test.logger",
                path="/tmp/grid.terraform",
            ),
        ),
    ]


def test_cluster_logs_reader_pagination():
    logs_api_client = mock.Mock()
    log_socket = mock.Mock()

    messages = iter(
        [
            r"""
            {
              "message": "v2",
              "timestamp": "2022-08-30T00:57:59.370356800Z",
              "labels": {
                "cluster_id": "cluster_id",
                "grid_url": "https://lightning.ai",
                "hostname": "ec2-001",
                "level": "info",
                "logger": "test.logger",
                "path": "/tmp/grid.terraform"
              }
            }
            """,
            r"""
            {
              "message": "v3",
              "timestamp": "2022-08-30T00:58:59.370356800Z",
              "labels": {
                "cluster_id": "cluster_id",
                "grid_url": "https://lightning.ai",
                "hostname": "ec2-001",
                "level": "info",
                "logger": "test.logger",
                "path": "/tmp/grid.terraform"
              }
            }
            """,
        ]
    )

    def create_cluster_logs_socket(
        cluster_id: str,
        start: int,  # unix timestamp
        end: int,  # unix timestamp
        limit: int,
        on_message_callback,
        on_error_callback,
    ):
        def run_forever():
            on_message_callback(None, next(messages))

        log_socket.run_forever = run_forever
        return log_socket

    logs_api_client.create_cluster_logs_socket = mock.Mock(
        side_effect=create_cluster_logs_socket,
    )

    logs = list(
        _cluster_logs_reader(
            logs_api_client=logs_api_client,
            cluster_id="cluster_id",
            start=0,
            end=10,
            limit=5,
            follow=False,
            batch_size=1,
        )
    )

    assert logs_api_client.create_cluster_logs_socket.call_args_list[0].kwargs["start"] == 0
    assert logs_api_client.create_cluster_logs_socket.call_args_list[0].kwargs["limit"] == 1
    assert logs_api_client.create_cluster_logs_socket.call_args_list[0].kwargs["end"] == 10

    assert logs_api_client.create_cluster_logs_socket.call_args_list[1].kwargs["start"] == 1661821079.370357
    assert logs_api_client.create_cluster_logs_socket.call_args_list[1].kwargs["limit"] == 1
    assert logs_api_client.create_cluster_logs_socket.call_args_list[1].kwargs["end"] == 10

    assert len(logs_api_client.create_cluster_logs_socket.call_args_list) == 3

    assert logs == [
        _ClusterLogEvent(
            message="v2",
            timestamp=datetime(2022, 8, 30, 0, 57, 59, 370356, tzinfo=tzutc()),
            labels=_ClusterLogEventLabels(
                cluster_id="cluster_id",
                grid_url="https://lightning.ai",
                hostname="ec2-001",
                level="info",
                logger="test.logger",
                path="/tmp/grid.terraform",
            ),
        ),
        _ClusterLogEvent(
            message="v3",
            timestamp=datetime(2022, 8, 30, 0, 58, 59, 370356, tzinfo=tzutc()),
            labels=_ClusterLogEventLabels(
                cluster_id="cluster_id",
                grid_url="https://lightning.ai",
                hostname="ec2-001",
                level="info",
                logger="test.logger",
                path="/tmp/grid.terraform",
            ),
        ),
    ]


def test_cluster_logs_limit_exceeded():
    logs_api_client = mock.Mock()
    log_socket = mock.Mock()

    log_message = r"""
        {
          "message": "v2",
          "timestamp": "2022-08-30T00:57:59.370356800Z",
          "labels": {
            "cluster_id": "cluster_id",
            "grid_url": "https://lightning.ai",
            "hostname": "ec2-001",
            "level": "info",
            "logger": "test.logger",
            "path": "/tmp/grid.terraform"
          }
        }
        """

    messages = None

    def create_cluster_logs_socket(
        cluster_id: str,
        start: int,  # unix timestamp
        end: int,  # unix timestamp
        limit: int,
        on_message_callback,
        on_error_callback,
    ):
        def run_forever():
            on_message_callback(None, next(messages))

        log_socket.run_forever = run_forever
        return log_socket

    logs_api_client.create_cluster_logs_socket = mock.Mock(
        side_effect=create_cluster_logs_socket,
    )

    messages = repeat(log_message, 2)
    _ = list(
        _cluster_logs_reader(
            logs_api_client=logs_api_client,
            cluster_id="cluster_id",
            start=0,
            end=10,
            limit=3,
            follow=False,
            batch_size=1,
        )
    )

    messages = repeat(log_message, 2)
    with pytest.raises(LogLinesLimitExceeded):
        _ = list(
            _cluster_logs_reader(
                logs_api_client=logs_api_client,
                cluster_id="cluster_id",
                start=0,
                end=10,
                limit=2,
                follow=False,
                batch_size=1,
            )
        )

    messages = repeat(log_message, 2)
    with pytest.raises(LogLinesLimitExceeded):
        _ = list(
            _cluster_logs_reader(
                logs_api_client=logs_api_client,
                cluster_id="cluster_id",
                start=0,
                end=10,
                limit=1,
                follow=False,
                batch_size=1,
            )
        )


def test_parse_log_event():
    assert (
        _parse_log_event(
            r"""
    {
      "message": "getting file lock",
      "timestamp": "2022-08-30T00:57:59.370356800Z",
      "labels": {
        "cluster_id": "cluster_id",
        "grid_url": "https://lightning.ai",
        "hostname": "ec2-001",
        "level": "info",
        "logger": "test.logger",
        "path": "/tmp/grid.terraform"
      }
    }
    """
        )
        == [
            _ClusterLogEvent(
                message="getting file lock",
                timestamp=datetime(2022, 8, 30, 0, 57, 59, 370356, tzinfo=tzutc()),
                labels=_ClusterLogEventLabels(
                    cluster_id="cluster_id",
                    grid_url="https://lightning.ai",
                    hostname="ec2-001",
                    level="info",
                    logger="test.logger",
                    path="/tmp/grid.terraform",
                ),
            ),
        ]
    )
