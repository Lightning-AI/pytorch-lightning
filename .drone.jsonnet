/*
Copyright The PyTorch Lightning team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// https://github.com/drone/drone-jsonnet-config/blob/master/.drone.jsonnet

local pipeline(name, image) = {
  kind: "pipeline",
  type: "docker",
  name: name,
  steps: [
    {
      name: "testing",
      image: image,
      environment: {
        "SLURM_LOCALID": 0,
        "CODECOV_TOKEN": {
          from_secret: "codecov_token"
        },
        "MKL_THREADING_LAYER": "GNU",
        "HOROVOD_GPU_OPERATIONS": "NCCL",
        "HOROVOD_WITH_PYTORCH": 1,
        "HOROVOD_WITHOUT_TENSORFLOW": 1,
        "HOROVOD_WITHOUT_MXNET": 1,
        "HOROVOD_WITH_GLOO": 1,
        "HOROVOD_WITHOUT_MPI": 1,
      },
      commands: [
        "export PATH=$PATH:/root/.local/bin",
        "python --version",
        "pip install pip -U",
        "pip --version",
        "nvidia-smi",
        "apt-get update && apt-get install -y cmake",
        "pip install -r ./requirements/base.txt -q --upgrade-strategy only-if-needed",
        "pip install -r ./requirements/devel.txt -q --upgrade-strategy only-if-needed",
        "pip install -r ./requirements/examples.txt -q --upgrade-strategy only-if-needed",
        // "pip install -r ./requirements/docs.txt -q --upgrade-strategy only-if-needed",
        "pip list",
        "python -c 'import torch ; print(' & '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]) if torch.cuda.is_available() else 'only CPU')'",
        "coverage run --source pytorch_lightning -m py.test pytorch_lightning tests -v --durations=25", // --flake8",
        "python -m py.test benchmarks pl_examples -v --maxfail=2 --durations=0", // --flake8",
        // "cd docs; make doctest; make coverage",
        "coverage report",
        // see: https://docs.codecov.io/docs/merging-reports
        "codecov --token $CODECOV_TOKEN --flags=gpu,pytest --name='GPU-coverage' --env=linux --build $DRONE_BUILD_NUMBER --commit $DRONE_COMMIT",
        // "--build $DRONE_BUILD_NUMBER --branch $DRONE_BRANCH --commit $DRONE_COMMIT --tag $DRONE_TAG --pr $DRONE_PULL_REQUEST",
        // "- codecov --token $CODECOV_TOKEN --flags=gpu,pytest --build $DRONE_BUILD_NUMBER",
        "python tests/collect_env_details.py"
      ],
    },
  ],
  trigger: {
    branch: "master",
    event: [
      "push",
      "pull_request"
    ]
  },
  depends_on: if name == "torch-GPU-nightly" then ["torch-GPU"]
};

[
    pipeline("torch-GPU", "pytorchlightning/pytorch_lightning:base-cuda-py3.7-torch1.6"),
    pipeline("torch-GPU-nightly", "pytorchlightning/pytorch_lightning:base-cuda-py3.7-torch1.7"),
]
