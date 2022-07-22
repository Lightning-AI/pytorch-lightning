import json
from datetime import datetime
from rich.table import Table
from rich.text import Text
import arrow

from lightning_cloud.openapi.models import (
    V1ClusterState,
    Externalv1Cluster,
    V1ClusterType,
)

from lightning_app.cli.core import Formatable

class ClusterList(Formatable):
    def __init__(self, clusters: [Externalv1Cluster]):
        self.clusters = clusters

    def as_json(self) -> str:
        return json.dumps(self.clusters)

    def as_table(self) -> Table:
        table = Table("id", "name", "type", "status", "created", show_header=True, header_style="bold green")
        phases = {
            V1ClusterState.QUEUED: Text("queued", style="bold yellow"),
            V1ClusterState.PENDING: Text("pending", style="bold yellow"),
            V1ClusterState.RUNNING: Text("running", style="bold green"),
            V1ClusterState.FAILED: Text("failed", style="bold red"),
            V1ClusterState.DELETED: Text("deleted", style="bold red"),
        }

        cluster_type_lookup = {
            V1ClusterType.BYOC: Text("byoc", style="bold yellow"),
            V1ClusterType.GLOBAL: Text("grid-cloud", style="bold green"),
        }
        for cluster in self.clusters:
            cluster: Externalv1Cluster
            status = phases[cluster.status.phase]
            if cluster.spec.desired_state == V1ClusterState.DELETED \
                    and cluster.status.phase != V1ClusterState.DELETED:
                status = Text("terminating", style="bold red")

            # this guard is necessary only until 0.3.93 releases which includes the `created_at`
            # field to the external API
            created_at = datetime.now()
            if hasattr(cluster, "created_at"):
                created_at = cluster.created_at

            table.add_row(
                cluster.id,
                cluster.name,
                cluster_type_lookup.get(cluster.spec.cluster_type, Text("unknown", style="red")),
                status,
                arrow.get(created_at).humanize() if created_at else "",
            )
        return table