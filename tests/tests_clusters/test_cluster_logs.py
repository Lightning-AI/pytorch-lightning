import os
import random
import string

import pytest

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
