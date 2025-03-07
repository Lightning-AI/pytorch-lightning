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


def main() -> None:
    """CLI entry point."""
    from fire import Fire  # type: ignore[import-untyped]

    Fire({
        "requirements": {
            "prune-pkgs": prune_packages_in_requirements,
            "set-oldest": replace_oldest_version,
            "replace-pkg": replace_package_in_requirements,
        },
        "version": lambda: print(lightning_utilities.__version__),
    })


if __name__ == "__main__":
    main()
