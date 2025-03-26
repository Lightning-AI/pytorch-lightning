# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Simple script to inject a custom JS script into all HTML pages in given folder.

Sample usage:
$ python scripts/inject-selector-script.py "/path/to/folder" torchmetrics

"""

import logging
import os
import sys


def inject_selector_script_into_html_file(file_path: str, script_url: str) -> None:
    """Inject a custom JS script into the given HTML file."""
    with open(file_path) as fopen:
        html_content = fopen.read()
    html_content = html_content.replace(
        "</head>",
        f'<script src="{script_url}" crossorigin="anonymous" referrerpolicy="no-referrer"></script>{os.linesep}</head>',
    )
    with open(file_path, "w") as fopen:
        fopen.write(html_content)


def main(folder: str, selector_name: str) -> None:
    """Inject a custom JS script into all HTML files in the given folder."""
    # Sample: https://lightning.ai/docs/torchmetrics/version-selector.js
    script_url = f"https://lightning.ai/docs/{selector_name}/version-selector.js"
    html_files = [
        os.path.join(root, file) for root, _, files in os.walk(folder) for file in files if file.endswith(".html")
    ]
    for file_path in html_files:
        inject_selector_script_into_html_file(file_path, script_url)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        from jsonargparse import auto_cli, set_parsing_settings

        set_parsing_settings(parse_optionals_as_positionals=True)
        auto_cli(main)
    except (ModuleNotFoundError, ImportError):
        main(*sys.argv[1:])
