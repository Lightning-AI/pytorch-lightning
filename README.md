# Lightning Sample project/package

This is starter project template which shall simplify initial steps for each new PL project...

[![UnitTests](https://github.com/Lightning-AI/dev-toolbox/actions/workflows/ci_testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/dev-toolbox/actions/workflows/ci_testing.yml)
[![Apply checks](https://github.com/Lightning-AI/dev-toolbox/actions/workflows/ci_use-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/dev-toolbox/actions/workflows/ci_use-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/pt-dev-toolbox/badge/?version=latest)](https://pt-dev-toolbox.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/dev-toolbox/main.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/Lightning-AI/dev-toolbox/main?badge_token=mqheL1-cTn-280Vx4cJUdg)

## To be Done

You still need to enable some external integrations such as:

- in GH setting, set `gh-pages` as website and _docs_ as source folder
- init Read-The-Docs (add this new project)
- add credentials for releasing package to PyPI

## Tests / Docs notes

- We are using [Napoleon style](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html), and we shall use static types...
- It is nice to se [doctest](https://docs.python.org/3/library/doctest.html) as they are also generated as examples in documentation
- For wider and edge cases testing use [pytest parametrization](https://docs.pytest.org/en/stable/parametrize.html) :\]
