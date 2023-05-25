<!-- Note: This document cannot be in `.github/README.md` because it will overwrite the repo README.md -->

# Continuous Integration and Delivery

## Unit and Integration Testing

| workflow file                          | action                                                                                                                                                                      | accelerator\* |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| .github/workflows/ci-tests-pytorch.yml | Run all tests except for accelerator-specific, standalone and slow tests.                                                                                                   | CPU           |
| .azure-pipelines/ipu-tests.yml         | Run only IPU-specific tests.                                                                                                                                                | IPU           |
| .azure-pipelines/gpu-tests-pytorch.yml | Run all CPU and GPU-specific tests, standalone, and examples. Each standalone test needs to be run in separate processes to avoid unwanted interactions between test cases. | GPU           |
| .azure-pipelines/gpu-benchmarks.yml    | Run speed/memory benchmarks for parity with pure PyTorch.                                                                                                                   | GPU           |
| .github/workflows/tpu-tests.yml        | Run only TPU-specific tests. Requires that the PR title contains '\[TPU\]'                                                                                                  | TPU           |

- \*Accelerators used in CI

  - GPU: 2 x NVIDIA RTX 3090
  - TPU: [Google TPU v4-8](https://cloud.google.com/tpu/docs/v4-users-guide)
  - IPU: [Colossus MK1 IPU](https://www.graphcore.ai/products/ipu)

- To check which versions of Python or PyTorch are used for testing in our CI, see the corresponding workflow files or checkgroup cofig file at [`.github/checkgroup.yml`](../checkgroup.yml).

## Documentation

| workflow file                     | action       |
| --------------------------------- | ------------ |
| .github/workflows/docs-checks.yml | Run doctest. |

## Code Quality

| workflow file                           | action                                                                                    |
| --------------------------------------- | ----------------------------------------------------------------------------------------- |
| .codecov.yml                            | Measure test coverage with [codecov.io](https://app.codecov.io/gh/Lightning-AI/lightning) |
| .github/workflows/code-checks.yml       | Check Python typing with [MyPy](https://mypy.readthedocs.io/en/stable/).                  |
| .github/workflows/ci-schema.yml         | Validate the syntax of workflow files.                                                    |
| .github/workflows/ci-check-md-links.yml | Validate links in markdown files.                                                         |

## Others

| workflow file                            | action                                                                                                                                                         |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| .github/workflows/ci-dockers-pytorch.yml | Build docker images used for testing in CI. If run on nightly schedule, push to the [Docker Hub](https://hub.docker.com/r/pytorchlightning/pytorch_lightning). |
| .github/workflows/ci-pkg-install.yml     | Test if pytorch-lightning is successfully installed using pip.                                                                                                 |

## Deployment

| workflow file                              | action                                                                             |
| ------------------------------------------ | ---------------------------------------------------------------------------------- |
| .github/workflows/release-pypi.yml         | Publish a release to PyPI.                                                         |
| .github/workflows/release-docker.yml       | Build Docker images from dockers/\*/Dockerfile and publish them on hub.docker.com. |
| .github/workflows/\_legacy-checkpoints.yml | App on request generate legacy checkpoints and upload them to AWS S3.              |

## Bots

| workflow file                                                      | action                                                                                                                                                                                                                  |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| .github/mergify.yml                                                | Label PRs as conflicts or ready, and request reviews if needed.                                                                                                                                                         |
| .github/stale.yml                                                  | Close inactive issues/PRs sometimes after adding the "won't fix" label to them.                                                                                                                                         |
| .github/workflows/probot-auto-cc.yml, .github/lightning-probot.yml | Notify maintainers of interest depending on labels added to an issue We utilize lightning-probot forked from PyTorch’s probot.                                                                                          |
| .github/workflows/probot-check-group.yml, .github/checkgroup.yml   | Checks whether the relevant jobs were successfully run based on the changed files in the PR                                                                                                                             |
| .pre-commit-config.yaml                                            | pre-commit.ci runs a set of linters and formatters, such as black, ruff and isort. When formatting is applied, the bot pushes a commit with its change. This configuration is also used for running pre-commit locally. |
| .github/workflows/labeler.yml, .github/labeler.yml                 | Integration of https://github.com/actions/labeler                                                                                                                                                                       |
