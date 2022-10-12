import argparse
import os
import re
from pathlib import Path
from typing import List, Optional

import requests
from pydantic import BaseModel

from lightning.app.storage.copier import copy_files
from lightning.app.storage.path import shared_storage_path
from lightning.app.utilities.commands import ClientCommand
from lightning.app.utilities.network import LightningClient
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.commands.artifacts.show import show_artifacts, ShowArtifactsConfig


class DownloadArtifactsConfigResponse(BaseModel):
    paths: List[str]
    urls: Optional[List[str]] = None


class DownloadArtifactsCommand(ClientCommand):
    def run(self) -> None:
        # 1. Parse the user arguments.
        parser = argparse.ArgumentParser()
        parser.add_argument("--components", nargs="+", default=[], help="Provide a list of component names.")
        parser.add_argument(
            "--output_dir", required=True, type=str, help="Provide the output directory for the artifacts.."
        )
        hparams = parser.parse_args()

        output_dir = Path(hparams.output_dir).resolve()

        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True)

        response = DownloadArtifactsConfigResponse(
            **self.invoke_handler(config=ShowArtifactsConfig(components=hparams.components, replace=False))
        )

        if response.urls:
            self._download_from_cloud(response, output_dir)
        else:
            self._download_local(response, output_dir)

    def _download_local(self, response: DownloadArtifactsConfigResponse, output_dir: Path):
        for path in response.paths:
            source_path = Path(path).resolve()
            target_file = Path(os.path.join(output_dir, path.split("artifacts/")[1])).resolve()
            if not target_file.parent.exists():
                os.makedirs(str(target_file.parent), exist_ok=True)
            copy_files(source_path, target_file)

    def _download_from_cloud(self, response: DownloadArtifactsConfigResponse, output_dir: Path):
        assert response.urls
        for path, url in zip(response.paths, response.urls):
            if path.startswith("/"):
                path = path[1:]
            resp = requests.get(url, allow_redirects=True)
            target_file = Path(os.path.join(output_dir, path.split("artifacts/")[1])).resolve()
            target_file.parent.mkdir(exist_ok=True, parents=True)
            with open(target_file, "wb") as f:
                f.write(resp.content)


def download_artifacts(config: ShowArtifactsConfig) -> DownloadArtifactsConfigResponse:
    """This function is responsible to collect the files from the the shared filesystem."""

    use_localhost = "LIGHTNING_APP_STATE_URL" not in os.environ

    if use_localhost:
        return DownloadArtifactsConfigResponse(paths=show_artifacts(config=config).paths)
    else:
        include = None

        if config.components:
            include = re.compile("|".join(config.components))

        client = LightningClient()
        project_id = _get_project(client).project_id
        app_id = os.getenv("LIGHTNING_CLOUD_APP_ID")
        response = client.lightningapp_instance_service_list_lightningapp_instance_artifacts(project_id, app_id)
        shared_storage = shared_storage_path()
        paths = [
            artifact.filename.replace(str(shared_storage), "").replace("/artifacts", "")
            for artifact in response.artifacts
            if include is None or include.match(artifact.filename)
        ]
        maps = {artifact.filename: artifact.url for artifact in response.artifacts}
        return DownloadArtifactsConfigResponse(paths=paths, urls=[maps[path] for path in paths])


DOWNLOAD_ARTIFACT = {"download artifacts": DownloadArtifactsCommand(download_artifacts)}
