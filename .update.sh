#!/bin/bash

version=$1

git commit -am "release v$version"
git tag $version -m "test_tube v$version"
git push --tags origin master

# push to pypi
rm -rf ./dist/*
python3 setup.py sdist
twine upload dist/*

# to update docs
# cd to root dir
# mkdocs gh-deploy

