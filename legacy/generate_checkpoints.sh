#!/bin/bash

VERSIONS=("1.0.0" "1.0.1" "1.0.2")

LEGACY_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

echo $LEGACY_PATH

for ver in "${VERSIONS[@]}"
do
	echo "---$ver---"
done
