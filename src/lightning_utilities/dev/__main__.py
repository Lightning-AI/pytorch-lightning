import fire

from lightning_utilities.dev.dependencies import prune_pkgs_in_requirements, replace_oldest_ver


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
