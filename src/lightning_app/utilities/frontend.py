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


def update_index_file_with_info(ui_root: str, info: AppInfo = None) -> None:
    import shutil
    from pathlib import Path

    entry_file = Path(ui_root) / "index.html"
    original_file = Path(ui_root) / "index.original.html"

    if not original_file.exists():
        shutil.copyfile(entry_file, original_file)  # keep backup
    else:
        # revert index.html in case it was modified after creating original.html
        shutil.copyfile(original_file, entry_file)

    if not info:
        return

    original = ""

    with original_file.open() as f:
        original = f.read()

    with entry_file.open("w") as f:
        f.write(_get_updated_content(original=original, info=info))


def _get_updated_content(original: str, info: AppInfo) -> str:
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
        soup.find("head").append(*[BeautifulSoup(meta, "html.parser") for meta in info.meta_tags])

    return str(soup)
