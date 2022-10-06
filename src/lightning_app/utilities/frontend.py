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


def update_index_file_with_info_and_prefix(ui_root: str, prefix: str, info: AppInfo = None) -> None:
    import shutil
    from pathlib import Path

    def rewrite_static_with_prefix(content: str):
        return content.replace("/static", f"{prefix}/static")

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
            f.write(rewrite_static_with_prefix(_get_updated_content(original=original, prefix=prefix, info=info)))

    if prefix:
        prefix_without_slash = prefix.replace("/", "", 1) if prefix.startswith("/") else prefix
        src_dir = Path(ui_root)
        dst_dir = src_dir / prefix_without_slash

        if dst_dir.exists():
            shutil.rmtree(dst_dir, ignore_errors=True)
        # copy everything except the current prefix, this is to fix a bug if user specifies
        # /abc at first and then /abc/def, server don't start
        # ideally we should copy everything except custom prefixes that user passed.
        shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns(f"{prefix_without_slash}*"))


def _get_updated_content(original: str, prefix: str, info: AppInfo) -> str:
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

    if prefix:
        # this will be used by lightning app ui to add prefix to add requests
        soup.find("head").append(BeautifulSoup(f'<script>window.app_prefix="{prefix}"</script>', "html.parser"))

    return str(soup)
