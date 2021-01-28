#!/usr/bin/env bash

ROOT=$PWD

#  python -m pip install protobuf
#  git clone --recursive https://github.com/onnx/onnx.git
#  cd onnx
#  python setup.py bdist_wheel
#  pip install --upgrade dist/*.whl
#  cd $ROOT
#  rm -rf onnx


# https://github.com/microsoft/onnxruntime/blob/master/BUILD.md
git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime
export ONNX_ML=1
pip install setuptools wheel numpy

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    ./build.sh --config RelWithDebInfo --build_shared_lib --build_wheel --parallel
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    ./build.sh --config RelWithDebInfo --build_shared_lib --build_wheel --parallel --use_xcode
elif [[ "$OSTYPE" == "cygwin" ]]; then
    # POSIX compatibility layer and Linux environment emulation for Windows
    ./build.sh --config RelWithDebInfo --build_shared_lib --build_wheel --parallel
elif [[ "$OSTYPE" == "msys" ]]; then
    # Lightweight shell and GNU utilities compiled for Windows (part of MinGW)
    .\build.bat --config RelWithDebInfo --build_shared_lib --build_wheel --parallel
elif [[ "$OSTYPE" == "win32" ]]; then
    .\build.bat --config RelWithDebInfo --build_shared_lib --build_wheel --parallel
else
    echo $OSTYPE  # Unknown.
fi

find . -name "*.whl"
pip install --upgrade $(find . -name "*.whl")

cd $ROOT
rm -rf onnxruntime
