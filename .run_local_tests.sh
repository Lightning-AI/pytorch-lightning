rm -rf tests/save_dir*
coverage run --source pytorch_lightning -m py.test pytorch_lightning tests examples -v --doctest-modules