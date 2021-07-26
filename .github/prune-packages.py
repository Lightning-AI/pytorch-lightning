import sys
from pprint import pprint


def main(req_file: str, *pkgs):
    with open(req_file, 'r') as fp:
        lines = fp.readlines()

    for pkg in pkgs:
        lines = [ln for ln in lines if not ln.startswith(pkg)]
    pprint(lines)

    with open(req_file, 'w') as fp:
        fp.writelines(lines)


if __name__ == "__main__":
    main(*sys.argv[1:])
