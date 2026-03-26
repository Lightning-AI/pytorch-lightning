# PyTorch Lightning Tests

Most tests in PyTorch Lightning train a [BoringModel](https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/demos/boring_classes.py) under various trainer conditions (ddp, amp, etc.). Want to add a new test case and not sure how? [Talk to us on Discord!](https://discord.gg/VptPCZkGNa)

## Test directory layout

```
tests/
├── tests_pytorch/   # Tests for PyTorch Lightning (Trainer, LightningModule, callbacks, …)
├── tests_fabric/    # Tests for Lightning Fabric (low-level distributed primitives)
├── parity_pytorch/  # Output parity checks: validates results match vanilla PyTorch
├── parity_fabric/   # Output parity checks for Fabric
└── legacy/          # Backward-compatibility tests for checkpoints from old releases
```

Each `tests_pytorch/` and `tests_fabric/` subtree mirrors the corresponding source layout under `src/lightning/`.

## Running tests

### Quick setup

```bash
# Clone, install all dev dependencies, and configure pre-commit in one step
make setup
```

### Run the full test suite

```bash
make test
```

This runs both packages with coverage. Note: GPU and TPU tests are skipped automatically on machines without the required hardware.

### Run tests for one package

```bash
# PyTorch Lightning only
pytest tests/tests_pytorch/ -v

# Lightning Fabric only
pytest tests/tests_fabric/ -v
```

### Run a single file or test

```bash
# Single file
pytest tests/tests_pytorch/trainer/test_trainer.py -v

# Single test function
pytest tests/tests_pytorch/trainer/test_trainer.py::TestClass::test_method -v

# Match by keyword
pytest tests/tests_pytorch/ -k "test_trainer_fit" -v
```

### Run with coverage

```bash
# PyTorch Lightning
python -m coverage run --source src/lightning/pytorch -m pytest src/lightning/pytorch tests/tests_pytorch -v
python -m coverage report -m

# Lightning Fabric
python -m coverage run --source src/lightning/fabric -m pytest src/lightning/fabric tests/tests_fabric -v
python -m coverage report -m
```

## Conditional tests and the `RunIf` decorator

Hardware- or package-gated tests use the `RunIf` helper instead of a bare `pytest.mark.skipif`:

```python
from tests.helpers.runif import RunIf  # tests_pytorch
# or
from tests.helpers.runif import RunIf  # tests_fabric

@RunIf(min_cuda_gpus=1)
def test_something_that_needs_a_gpu():
    ...

@RunIf(min_cuda_gpus=2, standalone=True)
def test_distributed():
    ...
```

Common `RunIf` kwargs:

| Kwarg               | What it guards                       |
| ------------------- | ------------------------------------ |
| `min_cuda_gpus=N`   | at least N CUDA GPUs available       |
| `min_torch="2.1"`   | PyTorch >= version                   |
| `mps=True`          | Apple Silicon (MPS) backend          |
| `skip_windows=True` | skip on Windows                      |
| `standalone=True`   | marks test as standalone (see below) |
| `deepspeed=True`    | DeepSpeed installed                  |
| `bf16_cuda=True`    | GPU supports bfloat16                |

GPU tests require that the environment variable `RUN_ONLY_CUDA_TESTS=1` is set; this is done automatically by CI.

## Standalone tests

Some distributed tests must run in a separate process to avoid interference. Mark them with `@RunIf(standalone=True)`. To run them locally:

```bash
PL_RUN_STANDALONE_TESTS=1 pytest tests/tests_pytorch/ -v
```

The CI helper script can also be used:

```bash
cd tests/
bash tests_pytorch/run_standalone_tasks.sh
```

## Pytest markers

The only custom marker registered in `pyproject.toml` is:

- `cloud` — tests that require cloud infrastructure (run in CI only)

Because pytest is configured with `--strict-markers`, **any new marker must be registered** in the `[tool.pytest.ini_options] markers` list in `pyproject.toml` before it can be used. Unregistered markers cause a collection error.

## Legacy (backward-compatibility) tests

The `legacy/` directory tests that checkpoints from older PyTorch Lightning releases can still be loaded. To pull the archived checkpoints from the public AWS storage:

```bash
bash .actions/pull_legacy_checkpoints.sh
```

See [tests/legacy/README.md](legacy/README.md) for details on generating checkpoints for a new release.

## Docker

You can also run tests inside the [pytorch-lightning CUDA Docker image](https://hub.docker.com/r/pytorchlightning/pytorch_lightning/tags?name=cuda):

```bash
# PyTorch Lightning
python -m pytest src/lightning/pytorch tests/tests_pytorch -v

# Lightning Fabric
python -m pytest src/lightning/fabric tests/tests_fabric -v
```

## GitHub Actions

Each push to a fork triggers CI automatically. This is useful for testing against all required dependency versions without a local GPU.
