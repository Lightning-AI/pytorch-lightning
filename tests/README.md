# PyTorch-Lightning Tests

Most of the tests in PyTorch Lightning train a [BoringModel](https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/demos/boring_classes.py) under various trainer conditions (ddp, amp, etc...). Want to add a new test case and not sure how? [Talk to us!](https://www.pytorchlightning.ai/community)

## Running tests

**Local:** Testing your work locally will help you speed up the process since it allows you to focus on particular (failing) test-cases.
To setup a local development environment, install both local and test dependencies:

```bash
# clone the repo
git clone https://github.com/Lightning-AI/lightning.git
cd lightning

# install required dependencies
export PACKAGE_NAME=pytorch
python -m pip install ".[dev, examples]"
# install pre-commit (optional)
python -m pip install pre-commit
pre-commit install
```

Additionally, for testing backward compatibility with older versions of PyTorch Lightning, you also need to download all saved version-checkpoints from the public AWS storage. Run the following script to get all saved version-checkpoints:

```bash
bash .actions/pull_legacy_checkpoints.sh
```

Note: These checkpoints are generated to set baselines for maintaining backward compatibility with legacy versions of PyTorch Lightning. Details of checkpoints for back-compatibility can be found [here](https://github.com/Lightning-AI/lightning/blob/master/tests/legacy/README.md).

You can run the full test suite in your terminal via this make script:

```bash
make test
```

Note: if your computer does not have multi-GPU or TPU these tests are skipped.

**GitHub Actions:** For convenience, you can also use your own GHActions building which will be triggered with each commit.
This is useful if you do not test against all required dependency versions.

**Docker:** Another option is to utilize the [pytorch lightning cuda base docker image](https://hub.docker.com/repository/docker/pytorchlightning/pytorch_lightning/tags?page=1&name=cuda). You can then run:

```bash
python -m pytest src/lightning/pytorch tests/tests_pytorch -v
```

You can also run a single test as follows:

```bash
python -m pytest -v tests/tests_pytorch/trainer/test_trainer_cli.py::test_default_args
```

### Conditional Tests

To test models that require GPU make sure to run the above command on a GPU machine.
The GPU machine must have at least 2 GPUs to run distributed tests.

Note that this setup will not run tests that require specific packages installed
You can rely on our CI to make sure all these tests pass.

### Standalone Tests

There are certain standalone tests, which you can run using:

```bash
./tests/run_standalone_tests.sh tests/tests_pytorch/trainer/
# or run a specific test
./tests/run_standalone_tests.sh -k test_multi_gpu_model_ddp
```

## Running Coverage

Make sure to run coverage on a GPU machine with at least 2 GPUs.

```bash
# generate coverage (coverage is also installed as part of dev dependencies)
coverage run --source src/lightning/pytorch -m pytest src/lightning/pytorch tests/tests_pytorch -v

# print coverage stats
coverage report -m

# exporting results
coverage xml
```
