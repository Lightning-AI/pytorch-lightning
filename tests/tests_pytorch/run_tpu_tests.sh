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
for fpath in `ls requirements/**/*.txt`; do
  python3 requirements/pytorch/adjust-versions.py $fpath {PYTORCH_VERSION};
done
pip install .[pytorch-extra,pytorch-test] pytest-timeout
pip list

# https://cloud.google.com/tpu/docs/v4-users-guide#train_ml_workloads_with_pytorch_xla
export ALLOW_MULTIPLE_LIBTPU_LOAD=1
# XRT config
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export TPU_NUM_DEVICES=4

echo "--- Sanity check TPU availability ---"
python3 -c "import torch_xla; print(torch_xla)"
python3 -c "from lightning.pytorch.accelerators import TPUAccelerator; assert TPUAccelerator.is_available()"
echo "Sanity check passed!"

echo "--- Running PL tests ---"
cd tests/tests_pytorch
PL_RUN_TPU_TESTS=1 python3 -m coverage run --source=lightning -m pytest -vv --durations=0 --timeout 60 ./

echo "--- Running standalone PL tests ---"
PL_RUN_TPU_TESTS=1 PL_STANDALONE_TESTS_BATCH_SIZE=1 bash run_standalone_tests.sh

echo "--- Generating coverage ---"
python3 -m coverage xml
mv coverage.xml ~
