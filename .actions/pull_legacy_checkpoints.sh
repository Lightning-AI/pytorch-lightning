#!/bin/bash
# Run this script from the project root.
URL="https://pl-public-data.s3.amazonaws.com/legacy/checkpoints.zip"
mkdir -p test/legacy
# wget is simpler but does not work on Windows
python -c "from urllib.request import urlretrieve; urlretrieve('$URL', 'test/legacy/checkpoints.zip')"
ls -l test/legacy/
unzip -o test/legacy/checkpoints.zip -d test/legacy/
ls -l test/legacy/checkpoints/
