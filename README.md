# Devtools

[![UnitTests](https://github.com/Lightning-AI/devtools/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/devtools/actions/workflows/ci-testing.yml)
[![Apply checks](https://github.com/Lightning-AI/devtools/actions/workflows/ci-use-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/devtools/actions/workflows/ci-use-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-devtools/badge/?version=latest)](https://lightning-devtools.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/devtools/main.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/Lightning-AI/devtools/main?badge_token=mqheL1-cTn-280Vx4cJUdg)

__This repository covers the following use-cases:__

1. **GitHub workflows**
1. **GitHub actions**
1. **CLI `pl_devtools`**
1. **General Python utilities**

## 1. Reusable workflows

__Usage:__

```yaml
name: Check schema

on: [push]

jobs:
  check-schema:
    uses: Lightning-AI/devtools/.github/workflows/check-schema.yml@main
    with:
      azure-dir: ""
```

See usage of other workflows in [.github/workflows/ci_use-checks.yml](https://github.com/Lightning-AI/devtools/tree/main/.github/workflows/ci_use-checks.yml).

## 2. Reusable composite actions

See available composite actions [.github/actions/](https://github.com/Lightning-AI/devtools/tree/main/.github/actions).

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
    - uses: Lightning-AI/devtools/.github/actions/cache
      with:
        python-version: 3.9
        requires: oldest
        # requires: latest
```

## 3. CLI

The package provides common CLI commands.

<details>
  <summary>Installation</summary>
From source:

```bash
pip install https://github.com/Lightning-AI/devtools/archive/refs/heads/main.zip
```

From pypi:

```bash
pip install lightning-devtools
```

</details>

__Usage:__

```bash
python -m pl_devtools [group] [command]
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
$ python -m pl_devtools requirements set-oldest
$ cat requirements/test.txt
coverage==5.0
codecov==2.1
pytest==6.0
pytest-cov
pytest-timeout
```

</details>
