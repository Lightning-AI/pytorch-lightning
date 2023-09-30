# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import glob
import inspect
import os
import re
import sys
from typing import Tuple


def _transform_changelog(path_in: str, path_out: str) -> None:
    """Adjust changelog titles so not to be duplicated.

    Args:
        path_in: input MD file
        path_out: output also MD file

    """
    with open(path_in) as fp:
        chlog_lines = fp.readlines()
    # enrich short subsub-titles to be unique
    chlog_ver = ""
    for i, ln in enumerate(chlog_lines):
        if ln.startswith("## "):
            chlog_ver = ln[2:].split("-")[0].strip()
        elif ln.startswith("### "):
            ln = ln.replace("###", f"### {chlog_ver} -")
            chlog_lines[i] = ln
    with open(path_out, "w") as fp:
        fp.writelines(chlog_lines)


def _linkcode_resolve(domain: str, github_user: str, github_repo: str, info: dict) -> str:
    def find_source() -> Tuple[str, int, int]:
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        fname = str(inspect.getsourcefile(obj))
        # https://github.com/rtfd/readthedocs.org/issues/5735
        if any(s in fname for s in ("readthedocs", "rtfd", "checkouts")):
            # /home/docs/checkouts/readthedocs.org/user_builds/pytorch_lightning/checkouts/
            #  devel/pytorch_lightning/utilities/cls_experiment.py#L26-L176
            path_top = os.path.abspath(os.path.join("..", "..", ".."))
            fname = str(os.path.relpath(fname, start=path_top))
        else:
            # Local build, imitate master
            fname = f'master/{os.path.relpath(fname, start=os.path.abspath(".."))}'
        source, line_start = inspect.getsourcelines(obj)
        return fname, line_start, line_start + len(source) - 1

    if domain != "py" or not info["module"]:
        return ""
    try:
        filename = "%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    # import subprocess
    # tag = subprocess.Popen(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE,
    #                        universal_newlines=True).communicate()[0][:-1]
    branch = filename.split("/")[0]
    # do mapping from latest tags to master
    branch = {"latest": "master", "stable": "master"}.get(branch, branch)
    filename = "/".join([branch] + filename.split("/")[1:])
    return f"https://github.com/{github_user}/{github_repo}/blob/{filename}"
