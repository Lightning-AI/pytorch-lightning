if ! [ $READTHEDOCS_VERSION == "latest" -o $READTHEDOCS_VERSION == "stable" ];
then
    cd ./docs/source-pytorch ;
    PL_FAST_DOCS_DEV=1 make html --jobs 2 ;
    ls -lh ../build
else
    mkdir -p ./docs/build/html
fi