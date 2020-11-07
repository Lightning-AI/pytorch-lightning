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
        "CODECOV_TOKEN": {
          from_secret: "codecov_token"
        },
        "MKL_THREADING_LAYER": "GNU",
      },
      commands: [
        "python --version",
        "pip --version",
        "nvidia-smi",
        "pip install -r ./requirements/devel.txt --upgrade-strategy only-if-needed -v --no-cache-dir",
        "pip list",
        "coverage run --source pytorch_lightning -m pytest pytorch_lightning tests -v -ra --color=yes --durations=25",
        "python -m pytest benchmarks pl_examples -v -ra --color=yes --maxfail=2 --durations=0",
        "coverage report",
        "codecov --token $CODECOV_TOKEN --flags=gpu,pytest --name='GPU-coverage' --env=linux --build $DRONE_BUILD_NUMBER --commit $DRONE_COMMIT",
        "python tests/collect_env_details.py"
      ],
    },
  ],
  trigger: {
    branch: [
      "master",
      "release/*"
    ],
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
