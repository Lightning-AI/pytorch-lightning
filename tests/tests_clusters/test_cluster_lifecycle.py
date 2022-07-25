import pytest

from src.lightning_app.testing.testing import run_cli


@pytest.mark.cloud
def test_cluster_lifecycle() -> None:
    role_arn = "TODO"
    external_id = "TODO"
    region = "us-west-2"
    instance_types = "t2.small,t3.small"
    cluster_name = "TODO-randomize"
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
    with run_cli(["clusters", "list"]) as (stdout, stderr):
        assert "grid-cloud" in stdout, f"stdout: {stdout}\nstderr: {stderr}"
