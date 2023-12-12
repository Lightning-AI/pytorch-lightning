# building for PRs and skip stable and latest states
set -ex

if ! [ $READTHEDOCS_VERSION == "latest" -o $READTHEDOCS_VERSION == "stable" ];
then
    export FAST_DOCS_DEV=1 ;
    root=$(pwd) ;
    for pkg in 'app' 'fabric' 'pytorch' ;
    do
      cd $root/docs/source-$pkg ;
      make html --jobs $(nproc) ;
      cd $root/docs ;
      mv build/html build/$pkg ;
    done ;
    # cross-road
    rm -rf build/doctrees ;
    cp crossroad.html build/index.html
else
    echo "Void build... :-]" ;
    mkdir -p ./docs/build
    cp ./docs/redirect.html ./docs/build/index.html
fi
