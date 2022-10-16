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

      echo "--- Fetch the SHA's changes ---"
      git clone --single-branch --depth 1 https://github.com/Lightning-AI/lightning.git /home/runner/work/lightning
      cd home/runner/work/lightning
      git fetch origin --depth 1 pull/{PR_NUMBER}/head:test/{PR_NUMBER}
      git -c advice.detachedHead=false checkout {SHA}

      echo "--- Install PL ---"
      PACKAGE_NAME=pytorch FREEZE_REQUIREMENTS=1 pip install -e .[test]
      pip list

      echo $KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS
      export XRT_TPU_CONFIG="tpu_worker;0;${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS:7}"

      echo "--- Sanity check TPU availability ---"
      python -c "from lightning_lite.accelerators import TPUAccelerator; assert TPUAccelerator.is_available()"
      python -c "from pytorch_lightning.accelerators import TPUAccelerator; assert TPUAccelerator.is_available()"
      echo "Sanity check passed!"

      echo "--- Running Lite tests ---"
      cd tests/tests_lite
      PL_RUN_TPU_TESTS=1 coverage run --source=lightning_lite -m pytest -vv --durations=0 ./

      echo "--- Running standalone Lite tests ---"
      PL_STANDALONE_TESTS_SOURCE=lightning_lite PL_STANDALONE_TESTS_BATCH_SIZE=1 bash run_standalone_tests.sh

      echo "--- Running PL tests ---"
      cd ../tests_pytorch
      PL_RUN_TPU_TESTS=1 coverage run --source=pytorch_lightning -m pytest -vv --durations=0 ./

      echo "--- Running standalone PL tests ---"
      PL_STANDALONE_TESTS_SOURCE=pytorch_lightning PL_STANDALONE_TESTS_BATCH_SIZE=1 bash run_standalone_tests.sh

      echo "--- Generating coverage ---"
      coverage xml
      cat coverage.xml | tr -d '\t'
    |||
  ),
};

tputests.oneshotJob
