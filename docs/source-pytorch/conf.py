#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# import m2r
import glob
import os
import shutil
import sys
import warnings

import pt_lightning_sphinx_theme

import pytorch_lightning

# -----------------------
# VARIABLES WHEN WORKING ON DOCS... MAKE THIS TRUE TO BUILD FASTER
# -----------------------
_PL_FAST_DOCS_DEV = bool(int(os.getenv("PL_FAST_DOCS_DEV", 0)))

# -----------------------
# BUILD stuff
# -----------------------
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, "..", "..")
PATH_RAW_NB = os.path.join(PATH_ROOT, "_notebooks")
_SHOULD_COPY_NOTEBOOKS = True
sys.path.insert(0, os.path.abspath(PATH_ROOT))
sys.path.append(os.path.join(PATH_RAW_NB, ".actions"))

try:
    from assistant import AssistantCLI
except ImportError:
    _SHOULD_COPY_NOTEBOOKS = False
    warnings.warn("To build the code, please run: `git submodule update --init --recursive`", stacklevel=2)

FOLDER_GENERATED = "generated"
SPHINX_MOCK_REQUIREMENTS = int(os.environ.get("SPHINX_MOCK_REQUIREMENTS", True))

# -- Project documents -------------------------------------------------------
if _SHOULD_COPY_NOTEBOOKS:
    AssistantCLI.copy_notebooks(
        PATH_RAW_NB, PATH_HERE, "notebooks", patterns=[".", "course_UvA-DL", "lightning_examples"]
    )


def _transform_changelog(path_in: str, path_out: str) -> None:
    with open(path_in) as fp:
        chlog_lines = fp.readlines()
    # enrich short subsub-titles to be unique
    chlog_ver = ""
    for i, ln in enumerate(chlog_lines):
        if ln.startswith("## "):
            chlog_ver = ln[2:].split("-")[0].strip()
        elif ln.startswith("### "):
            ln = ln.replace("###", f"### {chlog_ver} -")
            chlog_lines[i] = ln
    with open(path_out, "w") as fp:
        fp.writelines(chlog_lines)


os.makedirs(os.path.join(PATH_HERE, FOLDER_GENERATED), exist_ok=True)
# copy all documents from GH templates like contribution guide
for md in glob.glob(os.path.join(PATH_ROOT, ".github", "*.md")):
    shutil.copy(md, os.path.join(PATH_HERE, FOLDER_GENERATED, os.path.basename(md)))
# copy also the changelog
_transform_changelog(
    os.path.join(PATH_ROOT, "src", "pytorch_lightning", "CHANGELOG.md"),
    os.path.join(PATH_HERE, FOLDER_GENERATED, "CHANGELOG.md"),
)

# -- Project information -----------------------------------------------------

project = "PyTorch Lightning"
copyright = pytorch_lightning.__copyright__
author = pytorch_lightning.__author__

# The short X.Y version
version = pytorch_lightning.__version__
# The full version, including alpha/beta/rc tags
release = pytorch_lightning.__version__

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.

needs_sphinx = "4.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    # 'sphinxcontrib.mockautodoc',  # raises error: directive 'automodule' is already registered ...
    # 'sphinxcontrib.fulltoc',  # breaks pytorch-theme with unexpected kw argument 'titles_only'
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.imgmath",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_paramlinks",
    "sphinx_togglebutton",
    "pt_lightning_sphinx_theme.extensions.lightning",
]

# Suppress warnings about duplicate labels (needed for PL tutorials)
suppress_warnings = [
    "autosectionlabel.*",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# https://berkeley-stat159-f17.github.io/stat159-f17/lectures/14-sphinx..html#conf.py-(cont.)
# https://stackoverflow.com/questions/38526888/embed-ipython-notebook-in-sphinx-document
# I execute the notebooks manually in advance. If notebooks test the code,
# they should be run at build time.
nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_requirejs_path = ""

# myst-parser, forcing to parse all html pages with mathjax
# https://github.com/executablebooks/MyST-Parser/issues/394
myst_update_mathjax = False

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_parsers = {".rst": "restructuredtext", ".txt": "markdown", ".md": "markdown", ".ipynb": "nbsphinx"}

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    f"{FOLDER_GENERATED}/PULL_REQUEST_TEMPLATE.md",
    "notebooks/sample-template*",
]

if _PL_FAST_DOCS_DEV:
    exclude_patterns.append("notebooks/*")
    exclude_patterns.append("tutorials.rst")


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# http://www.sphinx-doc.org/en/master/usage/theming.html#builtin-themes
# html_theme = 'bizstyle'
# https://sphinx-themes.org
html_theme = "pt_lightning_sphinx_theme"
html_theme_path = [pt_lightning_sphinx_theme.get_html_theme_path()]
# html_theme_path = ["/Users/williamfalcon/Developer/opensource/lightning_sphinx_theme"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "pytorch_project": "https://pytorchlightning.ai",
    "canonical_url": pytorch_lightning.__docs_url__,
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
}

html_logo = "_static/images/logo.svg"

html_favicon = "_static/images/icon.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_templates", "_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = project + "-doc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
    # Latex figure (float) alignment
    "figure_align": "htbp"
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [(master_doc, project + ".tex", project + " Documentation", author, "manual")]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, project, project + " Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        project,
        project + " Documentation",
        author,
        project,
        "One line description of project.",
        "Miscellaneous",
    )
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "torchmetrics": ("https://torchmetrics.readthedocs.io/en/stable/", None),
    "fairscale": ("https://fairscale.readthedocs.io/en/latest/", None),
    "graphcore": ("https://docs.graphcore.ai/en/latest/", None),
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


def setup(app):
    # this is for hiding doctest decoration,
    # see: http://z4r.github.io/python/2011/12/02/hides-the-prompts-and-output/
    app.add_js_file("copybutton.js")
    app.add_css_file("main.css")


# copy all notebooks to local folder
# path_nbs = os.path.join(PATH_HERE, 'notebooks')
# if not os.path.isdir(path_nbs):
#     os.mkdir(path_nbs)
# for path_ipynb in glob.glob(os.path.join(PATH_ROOT, 'notebooks', '*.ipynb')):
#     path_ipynb2 = os.path.join(path_nbs, os.path.basename(path_ipynb))
#     shutil.copy(path_ipynb, path_ipynb2)


# Ignoring Third-party packages
# https://stackoverflow.com/questions/15889621/sphinx-how-to-exclude-imports-in-automodule
def package_list_from_file(file):
    """List up package name (not containing version and extras) from a package list file."""
    mocked_packages = []
    with open(file) as fp:
        for ln in fp.readlines():
            # Example: `tqdm>=4.41.0` => `tqdm`
            # `[` is for package with extras
            found = [ln.index(ch) for ch in list(",=<>#[") if ch in ln]
            pkg = ln[: min(found)] if found else ln
            if pkg.rstrip():
                mocked_packages.append(pkg.rstrip())
    return mocked_packages


# define mapping from PyPI names to python imports
PACKAGE_MAPPING = {
    "Pillow": "PIL",
    "opencv-python": "cv2",
    "PyYAML": "yaml",
    "comet-ml": "comet_ml",
    "neptune-client": "neptune",
    "hydra-core": "hydra",
    "pyDeprecate": "deprecate",
}
MOCK_PACKAGES = []
if SPHINX_MOCK_REQUIREMENTS:
    _path_require = lambda fname: os.path.join(PATH_ROOT, "requirements", "pytorch", fname)
    # mock also base packages when we are on RTD since we don't install them there
    MOCK_PACKAGES += package_list_from_file(_path_require("base.txt"))
    MOCK_PACKAGES += package_list_from_file(_path_require("extra.txt"))
    MOCK_PACKAGES += package_list_from_file(_path_require("loggers.txt"))
    MOCK_PACKAGES += package_list_from_file(_path_require("strategies.txt"))
MOCK_PACKAGES = [PACKAGE_MAPPING.get(pkg, pkg) for pkg in MOCK_PACKAGES]

autodoc_mock_imports = MOCK_PACKAGES

autosummary_generate = True

autodoc_member_order = "groupwise"

autoclass_content = "both"

autodoc_default_options = {
    "members": True,
    "methods": True,
    "special-members": "__call__",
    "exclude-members": "_abc_impl",
    "show-inheritance": True,
}

# Sphinx will add “permalinks” for each heading and description environment as paragraph signs that
#  become visible when the mouse hovers over them.
# This value determines the text for the permalink; it defaults to "¶". Set it to None or the empty
#  string to disable permalinks.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_add_permalinks
html_permalinks = True
html_permalinks_icon = "¶"

# True to prefix each section label with the name of the document it is in, followed by a colon.
#  For example, index:Introduction for a section called Introduction that appears in document index.rst.
#  Useful for avoiding ambiguity when the same section heading appears in different documents.
# http://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html
autosectionlabel_prefix_document = True

# only run doctests marked with a ".. doctest::" directive
doctest_test_doctest_blocks = ""
doctest_global_setup = """
import importlib
import os
import sys
from typing import Optional

import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import (
    _APEX_AVAILABLE,
    _XLA_AVAILABLE,
    _TPU_AVAILABLE,
    _TORCHVISION_AVAILABLE,
    _TORCH_GREATER_EQUAL_1_10,
    _module_available,
)
_JSONARGPARSE_AVAILABLE = _module_available("jsonargparse")
"""
coverage_skip_undoc_in_source = True
