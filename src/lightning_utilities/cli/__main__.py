# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import fire

from lightning_utilities.cli.dependencies import prune_pkgs_in_requirements, replace_oldest_ver


def main() -> None:
    fire.Fire(
        {
            "requirements": {
                "prune-pkgs": prune_pkgs_in_requirements,
                "set-oldest": replace_oldest_ver,
            }
        }
    )


if __name__ == "__main__":
    main()
