set -e  # exit on error

echo "--- Install packages ---"
# show what's already installed
pip3 list
# typing-extensions==4.5.0 comes pre-installed in the environment, and pydantic doesn't support that, however,
# pip cannot upgrade it because it's in the system folder: needs sudo
sudo pip3 install -U typing-extensions
# set particular PyTorch version
pip3 install -q wget packaging
python3 -m wget https://raw.githubusercontent.com/Lightning-AI/utilities/main/scripts/adjust-torch-versions.py
for fpath in `ls requirements/**/*.txt`; do
  python3 adjust-torch-versions.py $fpath {PYTORCH_VERSION};
done
pip3 install .[fabric-test] pytest-timeout
pip3 list

# https://cloud.google.com/tpu/docs/v4-users-guide#train_ml_workloads_with_pytorch_xla
export ALLOW_MULTIPLE_LIBTPU_LOAD=1
export PJRT_DEVICE=TPU

echo "--- Sanity check TPU availability ---"
python3 -c "import torch_xla; print(torch_xla)"
python3 -c "from lightning.fabric.accelerators.xla import _XLA_AVAILABLE; print(str(_XLA_AVAILABLE))"
python3 -c "from lightning.fabric.accelerators import XLAAccelerator; assert XLAAccelerator.is_available()"
echo "Sanity check passed!"

echo "--- Running Fabric tests ---"
cd tests/tests_fabric
PL_RUN_TPU_TESTS=1 python3 -m coverage run --source=lightning -m pytest -vv --durations=0 --timeout 60 ./

echo "--- Running standalone Fabric tests ---"
PL_RUN_TPU_TESTS=1 PL_STANDALONE_TESTS_BATCH_SIZE=1 bash ../run_standalone_tests.sh "."

echo "--- Generating coverage ---"
python3 -m coverage xml
mv coverage.xml ~
