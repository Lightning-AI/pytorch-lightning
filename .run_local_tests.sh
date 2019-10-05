# use this to run tests
rm -rf tests/save_dir*
rm -rf tests/mlruns_*
coverage run --source pytorch_lightning -m py.test pytorch_lightning tests examples -v --doctest-modules   
coverage report -m
