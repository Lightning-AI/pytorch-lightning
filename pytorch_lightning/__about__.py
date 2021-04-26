import time

_this_year = time.strftime("%Y")
__version__ = '1.3.0rc2'
__author__ = 'William Falcon et al.'
__author_email__ = 'waf2107@columbia.edu'
__license__ = 'Apache-2.0'
__copyright__ = f'Copyright (c) 2018-{_this_year}, {__author__}.'
__homepage__ = 'https://github.com/PyTorchLightning/pytorch-lightning'
__docs_url__ = "https://pytorch-lightning.readthedocs.io/en/stable/"
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = (
    "PyTorch Lightning is the lightweight PyTorch wrapper for ML researchers."
    " Scale your models. Write less boilerplate."
)
__long_docs__ = """
Lightning is a way to organize your PyTorch code to decouple the science code from the engineering.
 It's more of a style-guide than a framework.

In Lightning, you organize your code into 3 distinct categories:

1. Research code (goes in the LightningModule).
2. Engineering code (you delete, and is handled by the Trainer).
3. Non-essential research code (logging, etc. this goes in Callbacks).

Although your research/production project might start simple, once you add things like GPU AND TPU training,
 16-bit precision, etc, you end up spending more time engineering than researching.
 Lightning automates AND rigorously tests those parts for you.

Overall, Lightning guarantees rigorously tested, correct, modern best practices for the automated parts.

Documentation
-------------
- https://pytorch-lightning.readthedocs.io/en/latest
- https://pytorch-lightning.readthedocs.io/en/stable
"""

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__homepage__",
    "__license__",
    "__version__",
]
