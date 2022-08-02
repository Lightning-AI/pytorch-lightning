import os
import uuid

import pytest

from src.lightning_app.testing.testing import run_cli


@pytest.mark.cloud
@pytest.mark.skipif(
    os.environ.get("LIGHTNING_BYOC_ROLE_ARN") is None, reason="missing LIGHTNING_BYOC_ROLE_ARN environment variable"
)
@pytest.mark.skipif(
    os.environ.get("LIGHTNING_BYOC_EXTERNAL_ID") is None,
    reason="missing LIGHTNING_BYOC_EXTERNAL_ID environment variable",
)
def test_cluster_lifecycle() -> None:
    role_arn = os.environ.get("LIGHTNING_BYOC_ROLE_ARN", None)
    external_id = os.environ.get("LIGHTNING_BYOC_EXTERNAL_ID", None)
    region = "us-west-2"
    instance_types = "t2.small,t3.small"
    cluster_name = "byoc-%s" % (uuid.uuid4())
    with run_cli(
        [
            "create",
            "cluster",
            cluster_name,
            "--provider",
            "aws",
            "--role-arn",
            role_arn,
            "--external-id",
            external_id,
            "--region",
            region,
            "--instance-types",
            instance_types,
            "--wait",
        ]
    ) as (stdout, stderr):
        assert "success" in stdout, f"stdout: {stdout}\nstderr: {stderr}"

    with run_cli(["list", "clusters"]) as (stdout, stderr):
        assert cluster_name in stdout, f"stdout: {stdout}\nstderr: {stderr}"

    with run_cli(["delete", "cluster", "--force", cluster_name]) as (stdout, stderr):
        assert "success" in stdout, f"stdout: {stdout}\nstderr: {stderr}"


@pytest.mark.cloud
def test_cluster_list() -> None:
    with run_cli(["list", "clusters"]) as (stdout, stderr):
        assert "lightning-cloud" in stdout, f"stdout: {stdout}\nstderr: {stderr}"
