local common = import 'common.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

local pl_tpu_tests = common.PyTorchTest {
    frameworkPrefix: 'pl',
    modelName: 'tpu-tests',
    mode: 'postsubmit',
    configMaps: [],

    timeout: 900, # 15 minutes, in seconds.

    tpuSettings+: {
      softwareVersion: 'pytorch-1.8',
    },
    imageTag: 'r1.8.1',

    accelerator: tpus.v3_8,

  command: utils.scriptCommand(
    |||
      cd pytorch-lightning
      coverage run --source=pytorch_lightning -m pytest -v --capture=no \
          pytorch_lightning/utilities/xla_device.py \
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

pl_tpu_tests.oneshotJob
