set -e  # exit on error

if [[ "{ACCELERATOR_TYPE}" = v5litepod* ]]; then
  sudo apt-get update -y
  sudo apt-get install libomp5 -y
  pip3 install mkl mkl-include
  pip3 install tf-nightly tb-nightly tbp-nightly
  pip3 install numpy
  sudo apt-get install numactl libopenblas-dev -y
  pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly%2B20230712-cp310-cp310-linux_x86_64.whl
  pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly%2B20230712-cp310-cp310-linux_x86_64.whl
  gsutil cp gs://cloud-tpu-tpuvm-artifacts/v5litepod-preview/pytorch/wheels/torchvision-0.16.0a0+fc838ad-cp310-cp310-linux_x86_64.whl .
  pip3 install torchvision-0.16.0a0+fc838ad-cp310-cp310-linux_x86_64.whl
  pip3 install torch_xla[tpuvm]
fi

echo "--- Install packages ---"
# show what's already installed
pip3 list
if [[ "{ACCELERATOR_TYPE}" != v5litepod* ]]; then
  # typing-extensions==4.5.0 comes pre-installed in the environment, and pydantic doesnt support that, however,
  # pip cannot upgrade it because it's in the system folder: needs sudo
  sudo pip3 install -U typing-extensions
  pip3 install -q wget packaging
  python3 -m wget https://raw.githubusercontent.com/Lightning-AI/utilities/main/scripts/adjust-torch-versions.py
  for fpath in `ls requirements/**/*.txt`; do
    python3 adjust-torch-versions.py $fpath {PYTORCH_VERSION};
  done
fi
pip3 install .[pytorch-extra,pytorch-test] pytest-timeout
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
python3 -c "from lightning.pytorch.accelerators import XLAAccelerator; assert XLAAccelerator.is_available()"
echo "Sanity check passed!"

echo "--- Running PL tests ---"
cd tests/tests_pytorch
PL_RUN_TPU_TESTS=1 python3 -m coverage run --source=lightning -m pytest -vv --durations=0 --timeout 120 ./

echo "--- Running standalone PL tests ---"
PL_RUN_TPU_TESTS=1 PL_STANDALONE_TESTS_BATCH_SIZE=1 bash run_standalone_tests.sh

echo "--- Generating coverage ---"
python3 -m coverage xml
mv coverage.xml ~
