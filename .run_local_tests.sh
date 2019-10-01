rm -rf tests/save_dir*
rm -rf tests/mlruns_9964541/mlruns/
coverage run --source pytorch_lightning -m py.test pytorch_lightning tests examples -v --doctest-modules
