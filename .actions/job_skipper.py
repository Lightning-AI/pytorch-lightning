import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--changed_files", type=list, action="store", required=True)
hparams = parser.parse_args()
print(hparams.changed_files)
