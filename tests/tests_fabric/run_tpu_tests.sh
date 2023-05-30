set -e  # exit on error

echo "--- Cloning lightning repo ---"
git clone --single-branch --depth 1 https://github.com/Lightning-AI/lightning.git
cd lightning
# PR triggered it, check it out
if [ "{PR_NUMBER}" != "master" ]; then  # if PR number is set
  echo "--- Fetch the PR changes ---"
  git fetch origin --depth 1 pull/{PR_NUMBER}/head:test/{PR_NUMBER}
  echo "--- Checkout PR changes ---"
  git -c advice.detachedHead=false checkout {SHA}
fi

echo "--- Install packages ---"
# set particular PyTorch version
pip install -q wget packaging
python3 -m wget https://raw.githubusercontent.com/Lightning-AI/utilities/main/scripts/adjust-torch-versions.py
for fpath in `ls requirements/**/*.txt`; do
  python3 adjust-torch-versions.py $fpath {PYTORCH_VERSION};
done
pip install .[fabric-test] pytest-timeout
pip list

# https://cloud.google.com/tpu/docs/v4-users-guide#train_ml_workloads_with_pytorch_xla
export ALLOW_MULTIPLE_LIBTPU_LOAD=1
if [ "{RUNTIME}" = "xrt" ]; then
  export XRT_TPU_CONFIG="localservice;0;localhost:51011"
  export TPU_NUM_DEVICES=4
else
  export PJRT_DEVICE=TPU
fi

echo "--- Sanity check TPU availability ---"
python3 -c "import torch_xla; print(torch_xla)"
python3 -c "from lightning.fabric.accelerators.xla import _XLA_AVAILABLE; print(str(_XLA_AVAILABLE))"
python3 -c "from lightning.fabric.accelerators import XLAAccelerator; assert XLAAccelerator.is_available()"
echo "Sanity check passed!"

echo "--- Running Fabric tests ---"
cd tests/tests_fabric
PL_RUN_TPU_TESTS=1 python3 -m coverage run --source=lightning -m pytest -vv --durations=0 --timeout 60 ./

echo "--- Running standalone Fabric tests ---"
PL_RUN_TPU_TESTS=1 PL_STANDALONE_TESTS_BATCH_SIZE=1 bash run_standalone_tests.sh

echo "--- Generating coverage ---"
python3 -m coverage xml
mv coverage.xml ~
