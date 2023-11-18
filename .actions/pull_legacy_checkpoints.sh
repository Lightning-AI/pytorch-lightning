#!/bin/bash

# Run this script from the project root.
URL="https://pl-public-data.s3.amazonaws.com/legacy/checkpoints.zip"
mkdir -p tests/legacy
# wget is simpler but does not work on Windows
python -c "from urllib.request import urlretrieve; urlretrieve('$URL', 'tests/legacy/checkpoints.zip')"
ls -l tests/legacy/

unzip -o tests/legacy/checkpoints.zip -d tests/legacy/
ls -l tests/legacy/checkpoints/
