#!/usr/bin/env bash

ROOT=$PWD

git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ./build.sh --config RelWithDebInfo --build_shared_lib --parallel
elif [[ "$OSTYPE" == "darwin"* ]]; then
        ./build.sh --config RelWithDebInfo --build_shared_lib --parallel --use_xcode
elif [[ "$OSTYPE" == "cygwin" ]]; then
        # POSIX compatibility layer and Linux environment emulation for Windows
elif [[ "$OSTYPE" == "msys" ]]; then
        # Lightweight shell and GNU utilities compiled for Windows (part of MinGW)
elif [[ "$OSTYPE" == "win32" ]]; then
        .\build.bat --config RelWithDebInfo --build_shared_lib --parallel
else
        echo $OSTYPE  # Unknown.
fi

cd $ROOT
rm -rf onnxruntime
