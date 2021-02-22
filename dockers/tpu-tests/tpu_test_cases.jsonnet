local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import "templates/utils.libsonnet";

local tputests = base.BaseTest {
  frameworkPrefix: 'pl',
  modelName: 'tpu-tests',
  mode: 'postsubmit',
  configMaps: [],

  timeout: 900, # 15 minutes, in seconds.

  image: 'pytorchlightning/pytorch_lightning',
  imageTag: 'base-xla-py3.6-torch{PYTORCH_VERSION}',

  tpuSettings+: {
    softwareVersion: 'pytorch-{PYTORCH_VERSION}',
  },
  accelerator: tpus.v3_8,

  command: utils.scriptCommand(
    |||
      conda activate lightning
      mkdir -p /home/runner/work/pytorch-lightning && cd /home/runner/work/pytorch-lightning
      git clone https://github.com/PyTorchLightning/pytorch-lightning.git
      cd pytorch-lightning
      echo $PWD
      git ls-remote --refs origin
      git fetch origin "refs/pull/{PR_NUMBER}/head:pr/{PR_NUMBER}" && git checkout "pr/{PR_NUMBER}"
      pip install -e .
      coverage run --source=pytorch_lightning -m pytest -v --capture=no \
          pytorch_lightning/utilities/xla_device_utils.py \
          tests/accelerators/test_tpu_backend.py \
          tests/models/test_tpu.py
      test_exit_code=$?
      echo "\n||| END PYTEST LOGS |||\n"
      coverage xml
      cat coverage.xml | tr -d '\t'
      test $test_exit_code -eq 0
    |||
  ),
};

tputests.oneshotJob
