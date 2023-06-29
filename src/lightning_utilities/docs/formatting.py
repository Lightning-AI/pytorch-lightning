# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import glob
import os
import re


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
