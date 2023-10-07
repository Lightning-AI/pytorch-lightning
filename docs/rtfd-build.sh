# building for PRs and skip stable and latest states

if ! [ $READTHEDOCS_VERSION == "latest" -o $READTHEDOCS_VERSION == "stable" ];
then
    export FAST_DOCS_DEV=1 ;
    root=$(pwd) ;
    # build Fabric
    cd $root/docs/source-fabric ;
    make html --jobs $(nproc) ;
    cd $root/docs ;
    mv build/html build/fabric ;
    # build PyTorch
    cd $root/docs/source-pytorch ;
    make html --jobs $(nproc) ;
    cd $root/docs ;
    mv build/html build/pytorch ;
    # cross-road
    rm -rf build/doctrees ;
    cp crossroad.html build/index.html
else
    echo "Void build... :-]" ;
    mkdir -p ./docs/build
    cp ./docs/redirect.html ./docs/build/index.html
fi
