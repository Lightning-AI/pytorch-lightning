import argparse
import os
import re
from typing import Dict, List

import rich
from pydantic import BaseModel
from rich.color import ANSI_COLOR_NAMES

from lightning.app.storage.path import filesystem, shared_storage_path
from lightning.app.utilities.commands import ClientCommand


def _add_colors(filename: str) -> str:
    colors = list(ANSI_COLOR_NAMES)
    color = "magenta"

    if ".yaml" in filename:
        color = colors[1]

    elif ".ckpt" in filename:
        color = colors[2]

    elif "events.out.tfevents" in filename:
        color = colors[3]

    elif ".py" in filename:
        color = colors[4]

    elif ".png" in filename:
        color = colors[5]

    return f"[{color}]{filename}[/{color}]"


def _walk_folder(tree: List[str], sorted_directories: Dict[str, List[str]], root_path: str, depth: int):
    parents = {}
    for directory in sorted_directories:
        splits = directory.split("/")
        if depth == len(splits):
            for path in sorted_directories[directory]:
                tree.append(path)
        else:
            root_folder = "/".join(splits[1 : depth + 1])  # E203
            if root_folder not in parents:
                parents[root_folder] = []
            parents[root_folder].append(directory)

    if not parents:
        return

    for root_folder, directories in parents.items():
        folder = root_folder.split("/")[-1]
        root_folder = f":open_file_folder: /{root_folder}/"
        tree.append(f"[black]{root_folder}[/black]")
        _walk_folder(
            tree,
            {directory: sorted_directories[directory] for directory in directories},
            os.path.join(root_path, folder),
            depth + 1,
        )


def show_paths(paths: List[str]) -> None:
    """Recursively build a Tree with directory contents."""
    paths = sorted(paths)
    directories = {}
    tree = []
    for p in paths:
        directory = os.path.dirname(p)
        if directory not in directories:
            directories[directory] = []
        directories[directory].append(p)

    _walk_folder(tree, directories, "artifacts", 1)

    for path in tree:
        if not path.startswith("[black"):
            splits = path.split("/")
            path = f'   [black]{"/".join(splits[:-1])}[/black]/{_add_colors(splits[-1])}'
        rich.print(path)


class ShowArtifactsConfig(BaseModel):
    components: List[str]
    replace: bool = True


class ShowArtifactsResponse(BaseModel):
    paths: List[str]


class ShowArtifactsCommand(ClientCommand):
    def run(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--components", nargs="+", default=[], help="Provide a list of component names.")
        hparams = parser.parse_args()
        response = ShowArtifactsResponse(
            **self.invoke_handler(config=ShowArtifactsConfig(components=hparams.components))
        )
        show_paths(response.paths)


def show_artifacts(config: ShowArtifactsConfig) -> ShowArtifactsResponse:
    """This function is responsible to collecting the files from the shared filesystem."""
    fs = filesystem()
    paths = []
    include = None

    if config.components:
        include = re.compile("|".join(config.components))

    shared_storage = shared_storage_path()
    for root_dir, _, files in fs.walk(shared_storage):
        if config.replace:
            root_dir = str(root_dir).replace(str(shared_storage), "").replace("/artifacts", "")
        for f in files:
            path = os.path.join(str(root_dir), f)
            if include is None or include.match(path):
                paths.append(path)

    return ShowArtifactsResponse(paths=paths)


SHOW_ARTIFACT = {"show artifacts": ShowArtifactsCommand(show_artifacts)}
