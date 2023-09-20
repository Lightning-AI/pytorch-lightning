# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import glob
import inspect
import os
import shutil
import sys

import lai_sphinx_theme
from lightning_utilities.docs import fetch_external_assets

import lightning

_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_ROOT = os.path.realpath(os.path.join(_PATH_HERE, "..", ".."))
sys.path.insert(0, os.path.abspath(_PATH_ROOT))

_SPHINX_MOCK_REQUIREMENTS = int(os.environ.get("SPHINX_MOCK_REQUIREMENTS", True))
_FAST_DOCS_DEV = int(os.environ.get("FAST_DOCS_DEV", True))

# -- Project information -----------------------------------------------------

# this name shall match the project name in Github as it is used for linking to code
project = "lightning"
copyright = lightning.__copyright__
author = lightning.__author__

# The short X.Y version
version = lightning.__version__
# The full version, including alpha/beta/rc tags
release = lightning.__version__

# Options for the linkcode extension
# ----------------------------------
github_user = "Lightning-AI"
github_repo = project

# -- Project documents -------------------------------------------------------

if not _FAST_DOCS_DEV:
    fetch_external_assets(
        docs_folder=_PATH_HERE,
        assets_folder="_static/fetched-s3-assets",
        retrieve_pattern=r"https?://[-a-zA-Z0-9_]+\.s3\.[-a-zA-Z0-9()_\\+.\\/=]+"
    )

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.

needs_sphinx = "5.3"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx_toolbox.collapse",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.linkcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.imgmath",
    # 'sphinxcontrib.mockautodoc',  # raises error: directive 'automodule' is already registered ...
    # 'sphinxcontrib.fulltoc',  # breaks pytorch-theme with unexpected kw argument 'titles_only'
    "sphinxcontrib.video",
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_paramlinks",
    "sphinx_togglebutton",
    "sphinx.ext.githubpages",
    "lai_sphinx_theme.extensions.lightning",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# myst-parser, forcing to parse all html pages with mathjax
# https://github.com/executablebooks/MyST-Parser/issues/394
myst_update_mathjax = False
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html?highlight=anchor#auto-generated-header-anchors
myst_heading_anchors = 3

# https://berkeley-stat159-f17.github.io/stat159-f17/lectures/14-sphinx..html#conf.py-(cont.)
# https://stackoverflow.com/questions/38526888/embed-ipython-notebook-in-sphinx-document
# I execute the notebooks manually in advance. If notebooks test the code,
# they should be run at build time.
nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_requirejs_path = ""

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
# source_suffix = ['.rst', '.md', '.ipynb']
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
    ".ipynb": "nbsphinx",
}

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source-app directory, that match files and
# directories to ignore when looking for source-app files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "PULL_REQUEST_TEMPLATE.md",
    "**/README.md/*",
    "readme.md",
    "_templates",
    "code_samples/convert_pl_to_app/requirements.txt",
    "**/_static/*"
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "lai_sphinx_theme"
html_theme_path = [os.environ.get('LIT_SPHINX_PATH', lai_sphinx_theme.get_html_theme_path())]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "pytorch_project": lightning.__homepage__,
    "analytics_id": "G-D3Q2ESCTZR",
    "canonical_url": lightning.__homepage__,
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
}

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
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source-app start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, project + ".tex", project + " Documentation", author, "manual"),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source-app start file, name, description, authors, manual section).
man_pages = [(master_doc, project, project + " Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source-app start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        project,
        project + " Documentation",
        author,
        project,
        lightning.__docs__,
        "Miscellaneous",
    ),
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

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    # "numpy": ("https://docs.scipy.org/doc/numpy/", None),
}

nitpicky = True


nitpick_ignore = [
    ("py:class", "typing.Self"),
    # missing in generated API
    ("py:exc", "MisconfigurationException"),
    # TODO: generated list of all existing ATM, need to be fixed
    ('py:exc', 'ApiException'),
    ('py:class', 'BaseModel'),
    ('py:exc', 'LightningPlatformException'),
    ('py:class', 'forwarded'),
    ('py:class', 'lightning.app.api.http_methods.Delete'),
    ('py:class', 'lightning.app.api.http_methods.Get'),
    ('py:class', 'lightning.app.api.http_methods.HttpMethod'),
    ('py:class', 'lightning.app.api.http_methods.Post'),
    ('py:class', 'lightning.app.api.http_methods.Put'),
    ('py:class', 'lightning.app.components.python.TracerPythonScript'),
    ('py:func', 'lightning.app.pdb.set_trace'),
    ('py:class', 'lightning.app.runners.runtime.Runtime'),
    ('py:class', 'lightning.app.source_code.local.LocalSourceCodeDir'),
    ('py:class', 'lightning.app.storage.Path'),
    ('py:class', 'lightning.app.storage.payload._BasePayload'),
    ('py:class', 'lightning.app.structures.Dict'),
    ('py:class', 'lightning.app.structures.List'),
    ('py:class', 'lightning.app.testing.testing.LightningTestApp'),
    ('py:class', 'lightning.app.utilities.app_status.WorkStatus'),
    ('py:class', 'lightning.app.utilities.frontend.AppInfo'),
    ('py:class', 'lightning.app.utilities.packaging.app_config.AppConfig'),
    ('py:class', 'lightning.app.utilities.packaging.build_config.BuildConfig'),
    ('py:class', 'lightning.app.utilities.packaging.cloud_compute.CloudCompute'),
    ('py:class', 'lightning.app.utilities.proxies.WorkRunExecutor'),
    ('py:class', 'lightning.app.utilities.tracer.Tracer'),
    ('py:class', 'lightning_cloud.openapi.models.cloudspace_id_runs_body.CloudspaceIdRunsBody'),
    ('py:class', 'lightning_cloud.openapi.models.externalv1_lightningapp_instance.Externalv1LightningappInstance'),
    ('py:class', 'lightning_cloud.openapi.models.v1_cloud_space.V1CloudSpace'),
    ('py:class', 'lightning_cloud.openapi.models.v1_env_var.V1EnvVar'),
    ('py:class', 'lightning_cloud.openapi.models.v1_flowserver.V1Flowserver'),
    ('py:class', 'lightning_cloud.openapi.models.v1_lightning_auth.V1LightningAuth'),
    ('py:class', 'lightning_cloud.openapi.models.v1_lightning_run.V1LightningRun'),
    ('py:class', 'lightning_cloud.openapi.models.v1_lightningwork_drives.V1LightningworkDrives'),
    ('py:class', 'lightning_cloud.openapi.models.v1_membership.V1Membership'),
    ('py:class', 'lightning_cloud.openapi.models.v1_network_config.V1NetworkConfig'),
    ('py:class', 'lightning_cloud.openapi.models.v1_queue_server_type.V1QueueServerType'),
    ('py:class', 'lightning_cloud.openapi.models.v1_work.V1Work'),
    ('py:class', 'pydantic.main.BaseModel'),
    ('py:meth', 'transfer'),
]

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


def setup(app):
    # this is for hiding doctest decoration,
    # see: http://z4r.github.io/python/2011/12/02/hides-the-prompts-and-output/
    app.add_js_file("copybutton.js")
    app.add_css_file("main.css")


# copy all notebooks to local folder
path_nbs = os.path.join(_PATH_HERE, "notebooks")
if not os.path.isdir(path_nbs):
    os.mkdir(path_nbs)
for path_ipynb in glob.glob(os.path.join(_PATH_ROOT, "notebooks", "*.ipynb")):
    path_ipynb2 = os.path.join(path_nbs, os.path.basename(path_ipynb))
    shutil.copy(path_ipynb, path_ipynb2)

# copy all examples to local folder
path_examples = os.path.join(_PATH_HERE, "..", "examples")
if not os.path.isdir(path_examples):
    os.mkdir(path_examples)
for path_app_example in glob.glob(os.path.join(_PATH_ROOT, "examples", "app_*")):
    path_app_example2 = os.path.join(path_examples, os.path.basename(path_app_example))
    if not os.path.isdir(path_app_example2):
        shutil.copytree(path_app_example, path_app_example2, dirs_exist_ok=True)


# Ignoring Third-party packages
# https://stackoverflow.com/questions/15889621/sphinx-how-to-exclude-imports-in-automodule
def _package_list_from_file(file):
    list_pkgs = []
    with open(file) as fp:
        lines = fp.readlines()
    for ln in lines:
        found = [ln.index(ch) for ch in list(",=<>#") if ch in ln]
        pkg = ln[: min(found)] if found else ln
        if pkg.rstrip():
            list_pkgs.append(pkg.rstrip())
    return list_pkgs


# define mapping from PyPI names to python imports
PACKAGE_MAPPING = {
    "PyYAML": "yaml",
}
MOCK_PACKAGES = []
if _SPHINX_MOCK_REQUIREMENTS:
    # mock also base packages when we are on RTD since we don't install them there
    MOCK_PACKAGES += _package_list_from_file(os.path.join(_PATH_ROOT, "requirements.txt"))
MOCK_PACKAGES = [PACKAGE_MAPPING.get(pkg, pkg) for pkg in MOCK_PACKAGES]

autodoc_mock_imports = MOCK_PACKAGES


# Resolve function
# This function is used to populate the (source-app) links in the API
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        fname = inspect.getsourcefile(obj)
        # https://github.com/rtfd/readthedocs.org/issues/5735
        if any(s in fname for s in ("readthedocs", "rtfd", "checkouts")):
            # /home/docs/checkouts/readthedocs.org/user_builds/pytorch_lightning/checkouts/
            #  devel/pytorch_lightning/utilities/cls_experiment.py#L26-L176
            path_top = os.path.abspath(os.path.join("..", "..", ".."))
            fname = os.path.relpath(fname, start=path_top)
        else:
            # Local build, imitate master
            fname = "master/" + os.path.relpath(fname, start=os.path.abspath(".."))
        source, lineno = inspect.getsourcelines(obj)
        return fname, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    # import subprocess
    # tag = subprocess.Popen(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE,
    #                        universal_newlines=True).communicate()[0][:-1]
    branch = filename.split("/")[0]
    # do mapping from latest tags to master
    branch = {"latest": "master", "stable": "master"}.get(branch, branch)
    filename = "/".join([branch] + filename.split("/")[1:])
    return f"https://github.com/{github_user}/{github_repo}/blob/{filename}"


autosummary_generate = True

autodoc_member_order = "groupwise"
autoclass_content = "both"
# the options are fixed and will be soon in release,
#  see https://github.com/sphinx-doc/sphinx/issues/5459
autodoc_default_options = {
    "members": None,
    "methods": None,
    # 'attributes': None,
    "special-members": "__call__",
    "exclude-members": "_abc_impl",
    "show-inheritance": True,
    "private-members": True,
    "noindex": True,
}

# Sphinx will add “permalinks” for each heading and description environment as paragraph signs that
#  become visible when the mouse hovers over them.
# This value determines the text for the permalink; it defaults to "¶". Set it to None or the empty
#  string to disable permalinks.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_permalinks
# html_add_permalinks = "¶"
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

from lightning.app import LightningWork, LightningFlow, LightningApp, CloudCompute
from lightning.fabric.loggers.tensorboard import _TENSORBOARD_AVAILABLE, _TENSORBOARDX_AVAILABLE
"""
coverage_skip_undoc_in_source = True

# skip false positive linkcheck errors from anchors
linkcheck_anchors = False

# ignore all links in any CHANGELOG file
linkcheck_exclude_documents = [r"^(.*\/)*CHANGELOG.*$"]
