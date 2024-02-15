<!-- Note: This document cannot be in `.github/README.md` because it will overwrite the repo README.md -->

# Continuous Integration and Delivery

Brief description of all our automation tools used for boosting development performances.

## Unit and Integration Testing

| workflow file                          | action                                                                                    | accelerator |
| -------------------------------------- | ----------------------------------------------------------------------------------------- | ----------- |
| .github/workflows/ci-tests-app.yml     | Run all tests (may need internet connectivity).                                           | CPU         |
| .github/workflows/ci-tests-fabric.yml  | Run all tests except for accelerator-specific and standalone.                             | CPU         |
| .github/workflows/ci-tests-pytorch.yml | Run all tests except for accelerator-specific and standalone.                             | CPU         |
| .github/workflows/ci-tests-data.yml    | Run unit and integration tests with data pipelining.                                      | CPU         |
| .github/workflows/ci-tests-store.yml   | Run integration tests on uploading models to cloud.                                       | CPU         |
| .azure-pipelines/gpu-tests-fabric.yml  | Run only GPU-specific tests, standalone\*, and examples.                                  | GPU         |
| .azure-pipelines/gpu-tests-pytorch.yml | Run only GPU-specific tests, standalone\*, and examples.                                  | GPU         |
| .azure-pipelines/gpu-benchmarks.yml    | Run speed/memory benchmarks for parity with vanila PyTorch.                               | GPU         |
| .github/workflows/ci-examples-app.yml  | Run integration tests with App examples.                                                  | CPU         |
| .github/workflows/ci-flagship-apps.yml | Run end-2-end tests with full applications, including deployment to the production cloud. | CPU         |
| .github/workflows/ci-tests-pytorch.yml | Run all tests except for accelerator-specific, standalone and slow tests.                 | CPU         |
| .github/workflows/tpu-tests.yml        | Run only TPU-specific tests. Requires that the PR title contains '\[TPU\]'                | TPU         |

\* Each standalone test needs to be run in separate processes to avoid unwanted interactions between test cases.

- Accelerators used in CI

  - GPU: 2 x NVIDIA RTX 3090
  - TPU: [Google TPU v4-8](https://cloud.google.com/tpu/docs)

- To check which versions of Python or PyTorch are used for testing in our CI, see the corresponding workflow files or checkgroup config file at [`.github/checkgroup.yml`](../checkgroup.yml).

## Documentation

| workflow file                                                                   | action                                                                   |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| .github/workflows/docs-build.yml                                                | Run doctest, linkcheck and full HTML build.                              |
| .github/workflows/ci-rtfd.yml                                                   | Append link to the PR description with temporaty ReadTheDocs build docs. |
| .github/workflows/ci-check-md-links.yml <br> .github/markdown.links.config.json | Validate links in markdown files.                                        |

## Code Quality

| workflow file                     | action                                                                                    |
| --------------------------------- | ----------------------------------------------------------------------------------------- |
| .codecov.yml                      | Measure test coverage with [codecov.io](https://app.codecov.io/gh/Lightning-AI/lightning) |
| .github/workflows/code-checks.yml | Check Python typing with [MyPy](https://mypy.readthedocs.io/en/stable/).                  |
| .github/workflows/ci-schema.yml   | Validate the syntax of workflow files.                                                    |

## Others

| workflow file                        | action                                                                                          |
| ------------------------------------ | ----------------------------------------------------------------------------------------------- |
| .github/workflows/docker-build.yml   | Build docker images used for testing in CI. If run on nightly schedule, push to the Docker Hub. |
| .github/workflows/ci-pkg-install.yml | Test if pytorch-lightning is successfully installed using pip.                                  |
| .github/workflows/ci-checkpoints.yml | Build checkpoints that are will be tested on release to ensure backwards-compatibility          |

The published Docker Hub project is https://hub.docker.com/r/pytorchlightning/pytorch_lightning.

## Deployment

| workflow file                              | action                                                                         |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| .github/workflows/docs-build.yml           | Build the docs for each project and puch it to GCS with automatics deployment. |
| .github/workflows/docker-build.yml         | Build docker images used for releases and push them to the Docker Hub.         |
| .github/workflows/release-pkg.yml          | Publish a release to PyPI and upload to the GH release page as artifact.       |
| .github/workflows/\_legacy-checkpoints.yml | Add on request generate legacy checkpoints and upload them to AWS S3.          |

## Bots

| workflow file                                                          | action                                                                                                                                                   |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| .github/mergify.yml                                                    | Label PRs as conflicts or ready, and request reviews if needed.                                                                                          |
| .github/stale.yml                                                      | Close inactive issues/PRs sometimes after adding the "won't fix" label to them.                                                                          |
| .github/workflows/probot-auto-cc.yml <br> .github/lightning-probot.yml | Notify maintainers of interest depending on labels added to an issue We utilize lightning-probot forked from PyTorch’s probot.                           |
| .github/workflows/probot-check-group.yml <br> .github/checkgroup.yml   | Checks whether the relevant jobs were successfully run based on the changed files in the PR                                                              |
| .pre-commit-config.yaml                                                | It applies a set of linters and formatters and can be registered with your local dev. If needed [bot](https://pre-commit.ci/) pushc changes to each PRs. |
| .github/workflows/labeler-pr.yml, .github/label-change.yml             | Integration of https://github.com/actions/labeler                                                                                                        |
| .github/workflows/labeler-issue.yml                                    | Parse user provided `lightning` version and set it as label.                                                                                             |
