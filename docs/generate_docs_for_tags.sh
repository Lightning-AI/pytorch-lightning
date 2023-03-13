#!/bin/bash
# Usage:
# 1. Generate docs with one or more specified versions:
#    export PACKAGE_NAME=app
#    bash generate_docs_from_tags.sh 1.9.3 1.9.2 1.9.1 1.9.0
set -e

PATH_HERE=$(cd $(dirname $0); pwd -P)
PATH_ROOT=$PATH_HERE/..
PATH_ENV=$PATH_HERE/vEnv-docs
# export PACKAGE_NAME=app
export FREEZE_REQUIREMENTS=1
export PYTHONPATH=$(dirname $PATH_HERE)  # for `import tests_pytorch`

echo PATH_HERE: $PATH_HERE
echo PATH_ROOT: $PATH_ROOT
echo PATH_ENV: $PATH_ENV
echo PYTHONPATH: $PYTHONPATH

function build_docs {
  python --version
  pip --version

  cd $PATH_ROOT
  pip install -q setuptools wheel python-multipart
	pip install -e . --quiet -r requirements/$PACKAGE_NAME/docs.txt -f pypi -f https://download.pytorch.org/whl/cpu/torch_stable.html
  pip list

  cd $PATH_HERE
	cd /source-$PACKAGE_NAME && make html --jobs $(nproc)

  mv $PATH_HERE/build/html $PATH_HERE/builds-$PACKAGE_NAME/$tag
}

# iterate over all arguments assuming that each argument is version
for tag in "$@"
do
  echo processing version: $tag

  # Don't install/update anything before activating venv
  # to avoid breaking any existing environment.
  python -m venv $PATH_ENV
  source $PATH_ENV/bin/activate

  git checkout $tag
  # git pull --recurse-submodules

  build_docs > "building-$PACKAGE_NAME_${tag}.log"

  deactivate
  rm -rf $PATH_ENV
done
