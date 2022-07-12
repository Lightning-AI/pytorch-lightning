#!/bin/bash
# Run this script from the project root.
URL="https://pl-public-data.s3.amazonaws.com/legacy/checkpoints.zip"
mkdir -p legacy
# wget is simpler but does not work on Windows
python -c "from urllib.request import urlretrieve; urlretrieve('$URL', 'legacy/checkpoints.zip')"
ls -l legacy/
unzip -o legacy/checkpoints.zip -d legacy/
ls -l legacy/checkpoints/
