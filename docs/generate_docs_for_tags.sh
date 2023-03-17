#!/bin/bash
# Usage:
# 1. Generate docs with one or more specified versions:
#    $ export PACKAGE_NAME=app
#    $ bash docs/generate_docs_for_tags.sh 1.9.3 1.9.2 1.9.1 1.9.0
#    OR
#    $ PACKAGE_NAME=pytorch bash docs/generate_docs_for_tags.sh 1.8.6 1.8.5 1.8.4 1.8.3 1.8.2 1.8.1 1.8.0
set -e

PATH_ROOT=~/Desktop/builds
PATH_ENV=$PATH_ROOT/venv-docs-$PACKAGE_NAME
# export PACKAGE_NAME=app
export FREEZE_REQUIREMENTS=1

echo PATH_ROOT: $PATH_ROOT
echo PATH_ENV: $PATH_ENV
echo PYTHONPATH: $PYTHONPATH

function build_docs {
  python --version
  pip --version

  pip install -q setuptools wheel python-multipart
	pip install -e . -q -r requirements/$PACKAGE_NAME/docs.txt \
	  -f ../pypi -f https://download.pytorch.org/whl/cpu/torch_stable.html
  pip list

	cd docs/source-$PACKAGE_NAME
	make html --jobs=$(nproc)
	# make html SPHINXOPTS="-W --keep-going" --jobs=$(nproc)
	cd ../..

  mkdir -p $PATH_ROOT/docs-${PACKAGE_NAME}
  mv docs/build/html $PATH_ROOT/docs-${PACKAGE_NAME}/$tag
}

# iterate over all arguments assuming that each argument is version
for tag in "$@"
do
  echo processing version: $tag

  # Don't install/update anything before activating venv
  # to avoid breaking any existing environment.
  python -m venv $PATH_ENV
  source $PATH_ENV/bin/activate

  cd $PATH_ROOT
  git clone --single-branch --branch $tag --depth 1 --recurse-submodules \
    https://github.com/Lightning-AI/lightning.git
  cd lightning

  build_docs > "$PATH_ROOT/building-${PACKAGE_NAME}_${tag}.log"

  cd ..
  rm -rf lightning

  deactivate
  rm -rf $PATH_ENV
done
