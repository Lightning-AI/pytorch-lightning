import random

from lightning_cloud.openapi import V1ClusterType, V1ProjectClusterBinding
from lightning_cloud.openapi.rest import ApiException

from lightning_app.utilities.network import LightningClient


def _ensure_cluster_project_binding(client: LightningClient, project_id: str, cluster_id: str) -> None:
    cluster_bindings = client.projects_service_list_project_cluster_bindings(project_id=project_id)

    for cluster_binding in cluster_bindings.clusters:
        if cluster_binding.cluster_id != cluster_id:
            continue
        if cluster_binding.project_id == project_id:
            return

    client.projects_service_create_project_cluster_binding(
        project_id=project_id,
        body=V1ProjectClusterBinding(cluster_id=cluster_id, project_id=project_id),
    )


def _get_default_cluster(client: LightningClient, project_id: str) -> str:
    """This utility implements a minimal version of the cluster selection logic used in the cloud.

    TODO: This should be requested directly from the platform.
    """
    cluster_bindings = client.projects_service_list_project_cluster_bindings(project_id=project_id).clusters

    if not cluster_bindings:
        raise ValueError(f"No clusters are bound to the project {project_id}.")

    if len(cluster_bindings) == 1:
        return cluster_bindings[0].cluster_id

    clusters = []
    for cluster_binding in cluster_bindings:
        try:
            clusters.append(client.cluster_service_get_cluster(cluster_binding.cluster_id))
        except ApiException:
            # If we failed to get the cluster, ignore it
            continue

    # Filter global clusters
    clusters = [cluster for cluster in clusters if cluster.spec.cluster_type == V1ClusterType.GLOBAL]

    if len(clusters) == 0:
        raise RuntimeError(f"No clusters found on `{client.api_client.configuration.host}`.")

    return random.choice(clusters).id
