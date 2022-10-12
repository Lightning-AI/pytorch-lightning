<!-- Note: This document cannot be in `.github/README.md` because it will overwrite the repo README.md -->

# Continuous Integration and Delivery

## Unit and Integration Testing

| workflow name              | workflow file                              | action                                                                                                                                                                      | accelerator\* |
| -------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| Test PyTorch full          | .github/workflows/ci-pytorch-test-full.yml | Run all tests except for accelerator-specific, standalone and slow tests.                                                                                                   | CPU           |
| Test PyTorch slow          | .github/workflows/ci-pytorch-test-slow.yml | Run only slow tests. Slow tests usually need to spawn threads and cannot be speed up or simplified.                                                                         | CPU           |
| pytorch-lightning (IPUs)   | .azure-pipelines/ipu-tests.yml             | Run only IPU-specific tests.                                                                                                                                                | IPU           |
| pytorch-lightning (HPUs)   | .azure-pipelines/hpu-tests.yml             | Run only HPU-specific tests.                                                                                                                                                | HPU           |
| pytorch-lightning (GPUs)   | .azure-pipelines/gpu-tests-pytorch.yml     | Run all CPU and GPU-specific tests, standalone, and examples. Each standalone test needs to be run in separate processes to avoid unwanted interactions between test cases. | GPU           |
| PyTorchLightning.Benchmark | .azure-pipelines/gpu-benchmark.yml         | Run speed/memory benchmarks for parity with pure PyTorch.                                                                                                                   | GPU           |
| test-on-tpus               | .circleci/config.yml                       | Run only TPU-specific tests.                                                                                                                                                | TPU           |

- \*Accelerators used in CI

  - GPU: 2 x NVIDIA Tesla V100
  - TPU: Google GKE TPUv3
  - IPU: [Colossus MK1 IPU](https://www.graphcore.ai/products/ipu)
  - HPU: [Intel Habana Gaudi SYS-420GH-TNGR](https://www.supermicro.com/en/products/system/AI/4U/SYS-420GH-TNGR) which has 8 Gaudi accelerators

- To check which versions of Python or PyTorch are used for testing in our CI, see the corresponding workflow files or checkgroup cofig file at [`.github/checkgroup.yml`](../checkgroup.yml).

## Documentation

| workflow file                     | action       |
| --------------------------------- | ------------ |
| .github/workflows/docs-checks.yml | Run doctest. |

## Code Quality

| workflow file                     | action                                                                                    |
| --------------------------------- | ----------------------------------------------------------------------------------------- |
| .codecov.yml                      | Measure test coverage with [codecov.io](https://app.codecov.io/gh/Lightning-AI/lightning) |
| .github/workflows/code-checks.yml | Check Python typing with [MyPy](https://mypy.readthedocs.io/en/stable/).                  |
| .github/workflows/ci-schema.yml   | Validate the syntax of workflow files.                                                    |

## Others

| workflow file                              | action                                                                                                                                                         |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| .github/workflows/cicd-pytorch-dockers.yml | Build docker images used for testing in CI. If run on nightly schedule, push to the [Docker Hub](https://hub.docker.com/r/pytorchlightning/pytorch_lightning). |
| .github/workflows/ci-pkg-install.yml       | Test if pytorch-lightning is successfully installed using pip.                                                                                                 |
| .github/workflows/events-recurrent.yml     | Terminate TPU jobs that live more than one hour to avoid possible resource exhaustion due to hangs.                                                            |

## Deployment

| workflow file                            | action                                                                             |
| ---------------------------------------- | ---------------------------------------------------------------------------------- |
| .github/workflows/release-pypi.yml       | Publish a release to PyPI.                                                         |
| .github/workflows/release-docker.yml     | Build Docker images from dockers/\*/Dockerfile and publish them on hub.docker.com. |
| .github/workflows/legacy-checkpoints.yml | App on request generate legacy checkpoints and upload them to AWS S3.              |
| .github/workflows/events-nightly.yml     | Publish the package to TestPyPI. Publish Docker images on hub.docker.com.          |

## Bots

| workflow file                                                      | action                                                                                                                                                                                                                    |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| .github/mergify.yml                                                | Label PRs as conflicts or ready, and request reviews if needed.                                                                                                                                                           |
| .github/stale.yml                                                  | Close inactive issues/PRs sometimes after adding the "won't fix" label to them.                                                                                                                                           |
| .github/workflows/probot-auto-cc.yml, .github/lightning-probot.yml | Notify maintainers of interest depending on labels added to an issue We utilize lightning-probot forked from PyTorch’s probot.                                                                                            |
| .github/workflows/probot-check-group.yml, .github/checkgroup.yml   | Checks whether the relevant jobs were successfully run based on the changed files in the PR                                                                                                                               |
| .pre-commit-config.yaml                                            | pre-commit.ci runs a set of linters and formatters, such as black, flake8 and isort. When formatting is applied, the bot pushes a commit with its change. This configuration is also used for running pre-commit locally. |
