import os
import re
import shutil
from itertools import chain
from os.path import dirname, isfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pkg_resources

REQUIREMENT_FILES = {
    "pytorch": (
        "requirements/pytorch/base.txt",
        "requirements/pytorch/extra.txt",
        "requirements/pytorch/strategies.txt",
        "requirements/pytorch/examples.txt",
    ),
    "app": (
        "requirements/app/base.txt",
        "requirements/app/ui.txt",
        "requirements/app/cloud.txt",
    ),
    "lite": (
        "requirements/lite/base.txt",
        "requirements/lite/strategies.txt",
    ),
}
REQUIREMENT_FILES_ALL = list(chain(*REQUIREMENT_FILES.values()))


def _retrieve_files(directory: str, *ext: str) -> List[str]:
    all_files = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if not ext or any(os.path.split(fname)[1].lower().endswith(e) for e in ext):
                all_files.append(os.path.join(root, fname))

    return all_files


def _replace_imports(lines: List[str], mapping: List[Tuple[str, str]]) -> List[str]:
    """Replace imports of standalone package to lightning.

    >>> lns = [
    ...     "lightning_app",
    ...     "delete_cloud_lightning_apps",
    ...     "from lightning_app import",
    ...     "lightning_apps = []",
    ...     "lightning_app and pytorch_lightning are ours",
    ...     "def _lightning_app():",
    ...     ":class:`~lightning_app.core.flow.LightningFlow`"
    ... ]
    >>> mapping = [("lightning_app", "lightning.app"), ("pytorch_lightning", "lightning.pytorch")]
    >>> _replace_imports(lns, mapping)  # doctest: +NORMALIZE_WHITESPACE
    ['lightning.app', 'delete_cloud_lightning_apps', 'from lightning.app import', 'lightning_apps = []',\
    'lightning.app and lightning.pytorch are ours', 'def _lightning_app():',\
    ':class:`~lightning.app.core.flow.LightningFlow`']
    """
    out = lines[:]
    for source_import, target_import in mapping:
        for i, ln in enumerate(out):
            out[i] = re.sub(rf"([^_]|^){source_import}([^_\w]|$)", rf"\1{target_import}\2", ln)
    return out


def copy_replace_imports(
    source_dir: str, source_imports: List[str], target_imports: List[str], target_dir: Optional[str] = None
) -> None:
    """Copy package content with import adjustments."""
    print(f"Replacing imports: {locals()}")
    assert len(source_imports) == len(target_imports), (
        "source and target imports must have the same length, "
        f"source: {len(source_imports)}, target: {len(target_imports)}"
    )
    if target_dir is None:
        target_dir = source_dir

    ls = _retrieve_files(source_dir)
    for fp in ls:
        fp_new = fp.replace(source_dir, target_dir)
        _, ext = os.path.splitext(fp)
        if ext in (".png", ".jpg", ".ico"):
            os.makedirs(dirname(fp_new), exist_ok=True)
            if not isfile(fp_new):
                shutil.copy(fp, fp_new)
            continue
        elif ext in (".pyc",):
            continue
        # Try to parse everything else
        with open(fp, encoding="utf-8") as fo:
            try:
                lines = fo.readlines()
            except UnicodeDecodeError:
                # a binary file, skip
                print(f"Skipped replacing imports for {fp}")
                continue
        lines = _replace_imports(lines, list(zip(source_imports, target_imports)))
        os.makedirs(os.path.dirname(fp_new), exist_ok=True)
        with open(fp_new, "w", encoding="utf-8") as fo:
            fo.writelines(lines)


def create_mirror_package(source_dir: str, package_mapping: Dict[str, str]) -> None:
    # replace imports and copy the code
    mapping = package_mapping.copy()
    mapping.pop("lightning", None)  # pop this key to avoid replacing `lightning` to `lightning.lightning`
    for new, previous in mapping.items():
        copy_replace_imports(
            source_dir=os.path.join(source_dir, previous),
            # pytorch_lightning uses lightning_lite, so we need to replace all imports for all directories
            source_imports=list(mapping.values()),
            target_imports=[f"lightning.{new}" for new in mapping],
            target_dir=os.path.join(source_dir, "lightning", new),
        )


class AssistantCLI:
    @staticmethod
    def requirements_prune_pkgs(packages: Sequence[str], req_files: Sequence[str] = REQUIREMENT_FILES_ALL) -> None:
        """Remove some packages from given requirement files."""
        if isinstance(req_files, str):
            req_files = [req_files]
        for req in req_files:
            AssistantCLI._prune_packages(req, packages)

    @staticmethod
    def _prune_packages(req_file: str, packages: Sequence[str]) -> None:
        """Remove some packages from given requirement files."""
        path = Path(req_file)
        assert path.exists()
        text = path.read_text()
        lines = text.splitlines()
        final = []
        for line in lines:
            ln_ = line.strip()
            if not ln_ or ln_.startswith("#"):
                final.append(line)
                continue
            req = list(pkg_resources.parse_requirements(ln_))[0]
            if req.name not in packages:
                final.append(line)
        print(final)
        path.write_text("\n".join(final))

    @staticmethod
    def _replace_min(fname: str) -> None:
        req = open(fname, encoding="utf-8").read().replace(">=", "==")
        open(fname, "w", encoding="utf-8").write(req)

    @staticmethod
    def replace_oldest_ver(requirement_fnames: Sequence[str] = REQUIREMENT_FILES_ALL) -> None:
        """Replace the min package version by fixed one."""
        for fname in requirement_fnames:
            AssistantCLI._replace_min(fname)

    @staticmethod
    def copy_replace_imports(
        source_dir: str, source_import: str, target_import: str, target_dir: Optional[str] = None
    ) -> None:
        """Copy package content with import adjustments."""
        source_imports = source_import.strip().split(",")
        target_imports = target_import.strip().split(",")
        copy_replace_imports(source_dir, source_imports, target_imports, target_dir=target_dir)


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(AssistantCLI, as_positional=False)
