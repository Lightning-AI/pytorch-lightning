# Lightning Sample project/package

This is starter project template which shall simplify initial steps for each new PL project...

[![CI testing](https://github.com/PyTorchLightning/lightning-sandbox/workflows/CI%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/lightning-sandbox/actions?query=workflow%3A%22CI+testing%22)
![Check Code formatting](https://github.com/PyTorchLightning/lightning-sandbox/workflows/Check%20Code%20formatting/badge.svg?branch=master&event=push)
![Docs check](https://github.com/PyTorchLightning/lightning-sandbox/workflows/Docs%20check/badge.svg?branch=master&event=push)
[![Documentation Status](https://readthedocs.org/projects/pt-lightning-sandbox/badge/?version=latest)](https://pt-lightning-sandbox.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/PyTorchLightning/lightning-sandbox/master.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/PyTorchLightning/lightning-sandbox/master?badge_token=mqheL1-cTn-280Vx4cJUdg)

\* the Read-The-Docs is failing as this one leads to the public domain which requires the repo to be public too

## Included

Listing the implemented sections:

- sample package named `pl_sandbox`
- setting [CI](https://github.com/PyTorchLightning/lightning-sandbox/actions?query=workflow%3A%22CI+testing%22) for package and _tests_ folder
- setup/install package
- setting docs with Sphinx
- automatic PyPI release on GH release
- Docs deployd as [GH pages](https://pytorchlightning.github.io/lightning-sandbox)
- Makefile for building docs with `make docs` and run all tests `make test`

## To be Done

You still need to enable some external integrations such as:

- rename `pl_<sandbox>` to anu other name, simple find-replace shall work well
- update path used in the badges to the repository
- in GH setting lock the master breach - no direct push without PR
- in GH setting set `gh-pages` as website and _docs_ as source folder
- init Read-The-Docs (add this new project)
- add credentials for releasing package to PyPI
- specify license in `LICENSE` file and package init

## Tests / Docs notes

- We are using [Napoleon style](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) and we shall use static types...
- It is nice to se [doctest](https://docs.python.org/3/library/doctest.html) as they are also generated as examples in documentation
- For wider and edge cases testing use [pytest parametrization](https://docs.pytest.org/en/stable/parametrize.html) :\]
