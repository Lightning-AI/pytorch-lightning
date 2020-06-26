local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

local tputests = base.BaseTest {
  frameworkPrefix: 'pl',
  modelName: 'tpu-tests',
  mode: 'presubmit',

  timeout: 900, # 15 minutes, in seconds.

  image: std.extVar('image'),
  imageTag: std.extVar('image-tag'),

  tpuSettings+: {
    softwareVersion: 'pytorch-nightly',
  },
  accelerator: tpus.v3_8,

  command: [
    'py.test',
    'pytorch-lightning/tests/models/test_tpu.py',
    '-v',
  ]
};

tputests.oneshotJob
