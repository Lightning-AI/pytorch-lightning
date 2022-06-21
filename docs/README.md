# PyTorch-Lightning Docs

We are using Sphinx with Napoleon extension.
Moreover, we set Google style to follow with type convention.

- [Napoleon formatting with Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- [ReStructured Text (reST)](https://docs.pylonsproject.org/projects/docs-style-guide/)
- [Paragraph-level markup](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#paragraphs)

See following short example of a sample function taking one position string and optional

```python
from typing import Optional


def my_func(param_a: int, param_b: Optional[float] = None) -> str:
    """Sample function.

    Args:
        param_a: first parameter
        param_b: second parameter

    Return:
        sum of both numbers

    Example::

        >>> my_func(1, 2)
        3

    Note:
        If you want to add something.
    """
    p = param_b if param_b else 0
    return str(param_a + p)
```

## Building Docs

When updating the docs, make sure to build them first locally and visually inspect the html files in your browser for
formatting errors. In certain cases, a missing blank line or a wrong indent can lead to a broken layout.
Run these commands

```bash
git submodule update --init --recursive
pip install -e . -r requirements/docs.txt
make docs
```

and open `docs/build/html/index.html` in your browser.

When you send a PR the continuous integration will run tests and build the docs. You can access a preview of the html pages in the
_Artifacts_ tab in CircleCI when you click on the task named _build-Docs_ of _ci-tests_ at the bottom of the PR page.

Notes (Optional):

- You need to have LaTeX installed for rendering math equations. You can for example install TeXLive by doing one of the following:
  - on Ubuntu (Linux) run `apt-get install texlive` or otherwise follow the instructions on the TeXLive website
  - use the [RTD docker image](https://hub.docker.com/r/readthedocs/build)

## Developing docs

When developing the docs, building docs can be VERY slow locally because of the notebook tutorials.
To speed this up, enable this flag in before building docs:

```bash
# builds notebooks which is slow
export PL_FAST_DOCS_DEV=0

# fast notebook build which is fast
export PL_FAST_DOCS_DEV=1
```

## docs CSS/theme

To change the CSS theme of the docs, go [here](https://github.com/Lightning-AI/lightning_sphinx_theme).
Apologies in advance... this is a bit complex to build and requires basic understanding of javascript/npm.
