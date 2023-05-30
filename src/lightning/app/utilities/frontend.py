# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import List, Optional

from bs4 import BeautifulSoup


@dataclass
class AppInfo:
    title: Optional[str] = None
    favicon: Optional[str] = None
    description: Optional[str] = None
    image: Optional[str] = None
    # ensure the meta tags are correct or the UI might fail to load.
    meta_tags: Optional[List[str]] = None
    on_connect_end: Optional[str] = None


def update_index_file(ui_root: str, info: Optional[AppInfo] = None, root_path: str = "") -> None:
    import shutil
    from pathlib import Path

    entry_file = Path(ui_root) / "index.html"
    original_file = Path(ui_root) / "index.original.html"

    if not original_file.exists():
        shutil.copyfile(entry_file, original_file)  # keep backup
    else:
        # revert index.html in case it was modified after creating original.html
        shutil.copyfile(original_file, entry_file)

    if info:
        with original_file.open() as f:
            original = f.read()

        with entry_file.open("w") as f:
            f.write(_get_updated_content(original=original, root_path=root_path, info=info))

    if root_path:
        root_path_without_slash = root_path.replace("/", "", 1) if root_path.startswith("/") else root_path
        src_dir = Path(ui_root)
        dst_dir = src_dir / root_path_without_slash

        if dst_dir.exists():
            shutil.rmtree(dst_dir, ignore_errors=True)
        # copy everything except the current root_path, this is to fix a bug if user specifies
        # /abc at first and then /abc/def, server don't start
        # ideally we should copy everything except custom root_path that user passed.
        shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns(f"{root_path_without_slash}*"))


def _get_updated_content(original: str, root_path: str, info: AppInfo) -> str:
    soup = BeautifulSoup(original, "html.parser")

    # replace favicon
    if info.favicon:
        soup.find("link", {"rel": "icon"}).attrs["href"] = info.favicon

    if info.title is not None:
        soup.find("title").string = info.title

    if info.description:
        soup.find("meta", {"name": "description"}).attrs["content"] = info.description

    if info.image:
        soup.find("meta", {"property": "og:image"}).attrs["content"] = info.image

    if info.meta_tags:
        for meta in info.meta_tags:
            soup.find("head").append(BeautifulSoup(meta, "html.parser"))

    if root_path:
        # this will be used by lightning app ui to add root_path to add requests
        soup.find("head").append(BeautifulSoup(f'<script>window.app_prefix="{root_path}"</script>', "html.parser"))

    return str(soup).replace("/static", f"{root_path}/static")
