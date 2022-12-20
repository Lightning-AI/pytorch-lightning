# Lightning Utilities

[![UnitTests](https://github.com/Lightning-AI/utilities/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/utilities/actions/workflows/ci-testing.yml)
[![Apply checks](https://github.com/Lightning-AI/utilities/actions/workflows/ci-use-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/utilities/actions/workflows/ci-use-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning_utilities/badge/?version=latest)](https://lightning-tools.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/utilities/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/utilities/main)

__This repository covers the following use-cases:__

1. **GitHub workflows**
1. **GitHub actions**
1. **CLI `lightning_utilities.cli`**
1. **General Python utilities in `lightning_utilities.core`**

## 1. Reusable workflows

__Usage:__

```yaml
name: Check schema

on: [push]

jobs:
  check-schema:
    uses: Lightning-AI/utilities/.github/workflows/check-schema.yml@main
    with:
      azure-dir: ""
```

See usage of other workflows in [.github/workflows/ci_use-checks.yml](https://github.com/Lightning-AI/utilities/tree/main/.github/workflows/ci_use-checks.yml).

## 2. Reusable composite actions

See available composite actions [.github/actions/](https://github.com/Lightning-AI/utilities/tree/main/.github/actions).

__Usage:__

```yaml
name: Do something with cache

on: [push]

jobs:
  pytest:
    runs-on: ubuntu-20.04
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: 3.9
    - uses: Lightning-AI/utilities/.github/actions/cache
      with:
        python-version: 3.9
        requires: oldest
        # requires: latest
```

## 3. CLI `lightning_utilities.cli`

The package provides common CLI commands.

<details>
  <summary>Installation</summary>
From source:

```bash
pip install https://github.com/Lightning-AI/utilities/archive/refs/heads/main.zip
```

From pypi:

```bash
pip install lightning_utilities[cli]
```

</details>

__Usage:__

```bash
python -m lightning_utilities.cli [group] [command]
```

<details>
  <summary>Example for setting min versions</summary>

```console
$ cat requirements/test.txt
coverage>=5.0
codecov>=2.1
pytest>=6.0
pytest-cov
pytest-timeout
$ python -m lightning_utilities.cli requirements set-oldest
$ cat requirements/test.txt
coverage==5.0
codecov==2.1
pytest==6.0
pytest-cov
pytest-timeout
```

</details>

## 4. General Python utilities `lightning_utilities.core`

<details>
  <summary>Installation</summary>

From pypi:

```bash
pip install lightning_utilities
```

</details>

__Usage:__

Example for optional imports:

```python
from lightning_utilities.core.imports import module_available

if module_available("some_package.something"):
    from some_package import something
```
