# use this to run tests
rm -rf _ckpt_*
rm -rf tests/save_dir*
rm -rf tests/mlruns_*
rm -rf tests/tests/*
coverage run --source pytorch_lightning -m py.test pytorch_lightning tests pl_examples -v --doctest-modules
coverage report -m
