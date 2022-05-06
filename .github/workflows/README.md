<!-- Note: This document cannot be in `.github/README.md` because it will overwrite the repo README.md -->

# Continuous Integration and Delivery

## Unit and Integration Testing

| workflow name               | workflow file                       | action                                                                                                                                                                                      | accelerator\* | (Python, PyTorch)                                | OS                  |
| --------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------------------------------------------ | ------------------- |
| Test full                   | .github/workflows/ci_test-full.yml  | Run all tests except for accelerator-specific, standalone and slow tests.                                                                                                                   | CPU           | (3.7, 1.8), (3.7, 1.11), (3.9, 1.8), (3.9, 1.11) | linux, mac, windows |
| Test with Conda             | .github/workflows/ci_test-conda.yml | Same as ci_test-full.yml but with dependencies installed with conda.                                                                                                                        | CPU           | (3.8, 1.8), (3.8, 1.9), (3.8, 1.10), (3.9, 1.11) | linux               |
| Test slow                   | .github/workflows/ci_test-slow.yml  | Run only slow tests. Slow tests usually need to spawn threads and cannot be speed up or simplified.                                                                                         | CPU           | (3.7, 1.8)                                       | linux, mac, windows |
| PL.pytorch-lightning (IPUs) | .azure-pipelines/ipu-tests.yml      | Run only IPU-specific tests.                                                                                                                                                                | IPU           | (3.8, 1.9)                                       | linux               |
| PL.pytorch-lightning (HPUs) | .azure-pipelines/hpu-tests.yml      | Run only HPU-specific tests.                                                                                                                                                                | HPU           | (3.8, 1.10)                                      | linux               |
| PL.pytorch-lightning (GPUs) | .azure-pipelines/gpu-tests.yml      | Run GPU-specific tests as well as non-accelerator-specific, standalone and example tests. Each standalone test needs to run individually to avoid unwanted interactions between test cases. | GPU           | (3.7, 1.8)                                       | linux               |
| PyTorchLightning.Benchmark  | .azure-pipelines/gpu-benchmark.yml  | Run speed/memory benchmarks against pure PyTorch.                                                                                                                                           | GPU           | (3.7, 1.8)                                       | linux               |
| test-on-tpus                | .circleci/config.yml                | Run only TPU-specific tests.                                                                                                                                                                | TPU           | (3.7, 1.9)                                       | linux               |

- \*Accelerators used in CI
  - GPU: 2 x NVIDIA P100
  - TPU: Google GKE TPUv2/3
  - IPU: [Colossus MK1 IPU](https://www.graphcore.ai/products/ipu)
  - HPU: [AWS EC2 dl1.24xlarge](https://aws.amazon.com/ec2/instance-types/dl1/) which has 8 Gaudi accelerators

## Documentation

| workflow file                      | action                                                                                       |
| ---------------------------------- | -------------------------------------------------------------------------------------------- |
| .github/workflows/ci_test-base.yml | Validate code examples in docstrings in the package with pytest’s doctest.                   |
| .github/workflows/docs-checks.yml  | Run doctest, build documentation, and upload built docs to make them available as artifacts. |
| .circleci/config.yml (build-docs)  | Build docs and host them on output.circleci-artifacts.com for easy access to the built docs. |
| .github/workflows/docs-link.yml    | Provide a direct link to built docs on output.circleci-artifacts.com.                        |

## Code Quality

| workflow file                     | action                                 |
| --------------------------------- | -------------------------------------- |
| .codecov.yml                      | Measure test coverage with codecov.io. |
| .github/workflows/code-checks.yml | Check typing with mypy.                |
| .github/workflows/ci_schema.yml   | Validate the syntax of workflow files. |

## Others

| workflow file                          | action                                                                                                                                                                                                               |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| .github/workflows/ci_dockers.yml       | Build docker images used for testing in CI without pushing to the Docker Hub (pytorchlightning/pytorch_lightning). Publishing these built images takes place in .github/release-docker.yml which only runs in mater. |
| .github/workflows/ci_pkg-install.yml   | Test if pytorch-lightning is successfully installed using pip.                                                                                                                                                       |
| .github/workflows/events-recurrent.yml | Terminate TPU jobs that live more than one hour to avoid possible resource exhaustion due to hangs.                                                                                                                  |

## Deployment

| workflow file                            | action                                                                             |
| ---------------------------------------- | ---------------------------------------------------------------------------------- |
| .github/workflows/release-pypi.yml       | Publish a release to PyPI.                                                         |
| .github/workflows/release-docker.yml     | Build Docker images from dockers/\*/Dockerfile and publish them on hub.docker.com. |
| .github/workflows/legacy-checkpoints.yml | Generate legacy checkpoints and upload them to AWS S3.                             |
| .github/workflows/events-nightly.yml     | Publish the package to TestPyPI. Publish Docker images on hub.docker.com.          |

## Bots

| workflow file                                                      | action                                                                                                                                                                                                                    |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| .github/mergify.yml                                                | Label PRs as conflicts or ready, and request reviews if needed.                                                                                                                                                           |
| .github/stale.yml                                                  | Close inactive issues/PRs sometimes after adding the "won't fix" label to them.                                                                                                                                             |
| .github/workflows/probot-auto-cc.yml, .github/lightning-probot.yml | Notify maintainers of interest depending on labels added to an issue We utilize lightning-probot forked from PyTorch’s probot.                                                                                            |
| .pre-commit-config.yaml                                            | pre-commit.ci runs a set of linters and formatters, such as black, flake8 and isort. When formatting is applied, the bot pushes a commit with its change. This configuration is also used for running pre-commit locally. |
| .github/workflows/ci_pr-gatekeeper.yml                             | Prevent PRs from merging into master without any Grid.ai employees’ approval.                                                                                                                                             |

## Dashboards

- CI status in master (GitHub Actions only) hosted by the PyTorch team: https://hud2.pytorch.org/ci/PyTorchLightning/pytorch-lightning/master/
- Azure Pipelines (GPUs): https://dev.azure.com/PytorchLightning/pytorch-lightning/_pipeline/analytics/stageawareoutcome?definitionId=6&contextType=build
