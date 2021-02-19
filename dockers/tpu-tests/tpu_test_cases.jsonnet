local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import "templates/utils.libsonnet";

local tputests = base.BaseTest {
  frameworkPrefix: 'pl',
  modelName: 'tpu-tests',
  mode: 'postsubmit',
  configMaps: [],

  timeout: 900, # 15 minutes, in seconds.

  image: pytorchlightning/pytorch_lightning:base-xla-py3.6-torch{PYTORCH_VERSION},

  tpuSettings+: {
    softwareVersion: 'pytorch-{PYTORCH_VERSION}',
  },
  accelerator: tpus.v3_8,

  command: utils.scriptCommand(
    |||
      mkdir -p /home/runner/work/pytorch-lightning && cd /home/runner/work/pytorch-lightning
      git clone https://github.com/PyTorchLightning/pytorch-lightning.git
      cd pytorch-lightning
      echo $PWD
      git fetch --all
      git checkout {SHA}

      # drop horovod as it is not needed
      python -c "fname = 'pytorch-lightning/requirements/extra.txt' ; lines = [line for line in open(fname).readlines() if not line.startswith('horovod')] ; open(fname, 'w').writelines(lines)" && \
      # drop fairscale as it is not needed
      python -c "fname = 'pytorch-lightning/requirements/extra.txt' ; lines = [line for line in open(fname).readlines() if 'fairscale' not in line] ; open(fname, 'w').writelines(lines)" && \

      pip install -e .
      pip install -r pytorch-lightning/requirements/devel.txt --no-cache-dir

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
