# Lightning Sample project/package

This is starter project template which shall simplify initial steps for each new PL project...

[![CI testing](https://github.com/Lightning-AI/dev-toolbox/workflows/CI%20testing/badge.svg?branch=master&event=push)](https://github.com/Lightning-AI/dev-toolbox/actions?query=workflow%3A%22CI+testing%22)
![Check Code formatting](https://github.com/Lightning-AI/dev-toolbox/workflows/Check%20Code%20formatting/badge.svg?branch=master&event=push)
![Docs check](https://github.com/Lightning-AI/dev-toolbox/workflows/Docs%20check/badge.svg?branch=master&event=push)
[![Documentation Status](https://readthedocs.org/projects/pt-dev-toolbox/badge/?version=latest)](https://pt-dev-toolbox.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/dev-toolbox/master.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/Lightning-AI/dev-toolbox/master?badge_token=mqheL1-cTn-280Vx4cJUdg)

## To be Done

You still need to enable some external integrations such as:

- in GH setting, set `gh-pages` as website and _docs_ as source folder
- init Read-The-Docs (add this new project)
- add credentials for releasing package to PyPI
- specify license in `LICENSE` file and package init

## Tests / Docs notes

- We are using [Napoleon style](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html), and we shall use static types...
- It is nice to se [doctest](https://docs.python.org/3/library/doctest.html) as they are also generated as examples in documentation
- For wider and edge cases testing use [pytest parametrization](https://docs.pytest.org/en/stable/parametrize.html) :\]
