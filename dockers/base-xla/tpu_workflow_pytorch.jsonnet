local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import "templates/utils.libsonnet";

local tputests = base.BaseTest {
  frameworkPrefix: 'pl',
  modelName: 'tpu-tests',
  mode: 'postsubmit',
  configMaps: [],

  timeout: 6000, # 100 minutes, in seconds.

  image: 'pytorchlightning/pytorch_lightning',
  imageTag: 'base-xla-py{PYTHON_VERSION}-torch{PYTORCH_VERSION}',

  tpuSettings+: {
    softwareVersion: 'pytorch-{PYTORCH_VERSION}',
  },
  accelerator: tpus.v3_8,

  command: utils.scriptCommand(
    |||
      set +x  # turn off tracing, spammy
      set -e  # exit on error

      source ~/.bashrc
      conda activate lightning

      echo "--- Cloning lightning repo ---"
      git clone --single-branch --depth 1 https://github.com/Lightning-AI/lightning.git
      cd lightning
      # PR triggered it, check it out
      if [ -n "{PR_NUMBER}" ]; then  # if PR number is not empty
        echo "--- Fetch the PR changes ---"
        git fetch origin --depth 1 pull/{PR_NUMBER}/head:test/{PR_NUMBER}
        echo "--- Checkout PR changes ---"
        git -c advice.detachedHead=false checkout {SHA}
      fi

      echo "--- Install packages ---"
      # set particular PyTorch version
      for fpath in `ls requirements/**/*.txt`; do
        python requirements/pytorch/adjust-versions.py $fpath {PYTORCH_VERSION};
      done
      PACKAGE_NAME=pytorch pip install .[extra,test]
      pip list

      pip install -q -r .actions/requirements.txt
      python .actions/assistant.py copy_replace_imports --source_dir="./tests" \
          --source_import="lightning.fabric,lightning.pytorch" \
          --target_import="lightning_fabric,pytorch_lightning"

      echo $KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS
      export XRT_TPU_CONFIG="tpu_worker;0;${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS:7}"

      echo "--- Sanity check TPU availability ---"
      python -c "import torch_xla; print(torch_xla)"
      python -c "from pytorch_lightning.accelerators.tpu import _XLA_AVAILABLE; print(str(_XLA_AVAILABLE))"
      python -c "from pytorch_lightning.accelerators import TPUAccelerator; assert TPUAccelerator.is_available()"
      echo "Sanity check passed!"

      echo "--- Running PL tests ---"
      cd tests/tests_pytorch
      PL_RUN_TPU_TESTS=1 coverage run --source=pytorch_lightning -m pytest -vv --durations=0 ./

      echo "--- Running standalone PL tests ---"
      PL_RUN_TPU_TESTS=1 PL_STANDALONE_TESTS_SOURCE=pytorch_lightning PL_STANDALONE_TESTS_BATCH_SIZE=1 bash run_standalone_tests.sh

      echo "--- Generating coverage ---"
      coverage xml
      cat coverage.xml | tr -d '\t'
    |||
  ),
};

tputests.oneshotJob
