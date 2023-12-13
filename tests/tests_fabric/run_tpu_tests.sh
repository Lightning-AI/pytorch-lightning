set -e  # exit on error

if [[ "{PYTORCH_VERSION}" = "2.1" ]]; then
  pip install "torch=={PYTORCH_VERSION}" -f https://storage.googleapis.com/libtpu-releases/index.html
  pip install "torch_xla[tpu]=={PYTORCH_VERSION}" -f https://storage.googleapis.com/libtpu-releases/index.html
fi

echo "--- Install packages ---"
# show what's already installed
pip3 list
pip3 install .[fabric-test] pytest-timeout
if [[ "{ACCELERATOR_TYPE}" == v5litepod* ]]; then
  # xla requires google-api-core which itself requires protobuf 3.20.3
  # on the other hand, fabric tests install tensorboardX which requires protobuf>=4.22.3
  # since TPU tests don't care about tensorboardX, we can just ignore that: welcome to dependency hell
  pip3 install google-api-core
fi
pip3 list

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
