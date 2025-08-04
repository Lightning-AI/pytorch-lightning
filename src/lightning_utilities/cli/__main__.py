# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#

import lightning_utilities
from lightning_utilities.cli.dependencies import (
    prune_packages_in_requirements,
    replace_oldest_version,
    replace_package_in_requirements,
)


def _get_version() -> None:
    """Prints the version of the lightning_utilities package."""
    print(lightning_utilities.__version__)


def main() -> None:
    """CLI entry point."""
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(
        {
            "requirements": {
                "_help": "Manage requirements files.",
                "prune-pkgs": prune_packages_in_requirements,
                "set-oldest": replace_oldest_version,
                "replace-pkg": replace_package_in_requirements,
            },
            "version": _get_version,
        },
        as_positional=False,
    )


if __name__ == "__main__":
    main()
