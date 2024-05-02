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

import glob
import os
import shutil
import warnings
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType

import lai_sphinx_theme
from lightning_utilities.docs.formatting import _transform_changelog

import lightning

# -----------------------
# VARIABLES WHEN WORKING ON DOCS... MAKE THIS TRUE TO BUILD FASTER
# -----------------------
_SPHINX_MOCK_REQUIREMENTS = int(os.environ.get("SPHINX_MOCK_REQUIREMENTS", True))
_FAST_DOCS_DEV = int(os.getenv("FAST_DOCS_DEV", True))
_COPY_NOTEBOOKS = int(os.getenv("DOCS_COPY_NOTEBOOKS", not _FAST_DOCS_DEV))
_FETCH_S3_ASSETS = int(os.getenv("DOCS_FETCH_ASSETS", not _FAST_DOCS_DEV))
_PIN_RELEASE_VERSIONS = int(os.getenv("PIN_RELEASE_VERSIONS", not _FAST_DOCS_DEV))

# -----------------------
# BUILD stuff
# -----------------------
_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_ROOT = os.path.join(_PATH_HERE, "..", "..")
_PATH_RAW_NB = os.path.join(_PATH_ROOT, "_notebooks")
_PATH_RAW_NB_ACTIONS = os.path.join(_PATH_RAW_NB, ".actions")
_FOLDER_GENERATED = "generated"


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


assist_local = _load_py_module("assistant", os.path.join(_PATH_ROOT, ".actions", "assistant.py"))
if os.path.isdir(_PATH_RAW_NB_ACTIONS):
    assist_nb = _load_py_module("assistant", os.path.join(_PATH_RAW_NB_ACTIONS, "assistant.py"))
else:
    _COPY_NOTEBOOKS = False
    warnings.warn(
        "To build full docs you need to include also tutorials/notebooks from submodule code."
        " Please run: `git submodule update --init --recursive`",
        stacklevel=2,
    )


# -- Project documents -------------------------------------------------------

if _COPY_NOTEBOOKS:
    assist_nb.AssistantCLI.copy_notebooks(
        _PATH_RAW_NB,
        _PATH_HERE,
        "notebooks",
        patterns=[".", "course_UvA-DL", "lightning_examples"],
        # TODO(@aniketmaurya): Complete converting the missing items and add them back
        ignore=[
            # "course_UvA-DL/13-contrastive-learning",
            "lightning_examples/warp-drive",
        ],
    )


os.makedirs(os.path.join(_PATH_HERE, _FOLDER_GENERATED), exist_ok=True)
# copy all documents from GH templates like contribution guide
for md in glob.glob(os.path.join(_PATH_ROOT, ".github", "*.md")):
    shutil.copy(md, os.path.join(_PATH_HERE, _FOLDER_GENERATED, os.path.basename(md)))
# copy also the changelog
_transform_changelog(
    os.path.join(_PATH_ROOT, "src", "lightning", "fabric", "CHANGELOG.md"),
    os.path.join(_PATH_HERE, _FOLDER_GENERATED, "CHANGELOG.md"),
)

# Copy Accelerator docs
assist_local.AssistantCLI.pull_docs_files(
    gh_user_repo="Lightning-AI/lightning-Habana",
    target_dir="docs/source-pytorch/integrations/hpu",
    checkout="refs/tags/1.4.0",
)

# Copy strategies docs as single pages
assist_local.AssistantCLI.pull_docs_files(
    gh_user_repo="Lightning-Universe/lightning-Hivemind",
    target_dir="docs/source-pytorch/integrations/strategies",
    checkout="3b14f766200aff8fe7153be19a7bd92440dea3cf",  # this is post release version including moved overview page
    single_page="overview.rst",
)

if _FETCH_S3_ASSETS:
    from lightning_utilities.docs import fetch_external_assets

    fetch_external_assets(
        docs_folder=_PATH_HERE,
        assets_folder="_static/fetched-s3-assets",
        retrieve_pattern=r"https?://[-a-zA-Z0-9_]+\.s3\.[-a-zA-Z0-9()_\\+.\\/=]+",
    )

if _PIN_RELEASE_VERSIONS:
    from lightning_utilities.docs import adjust_linked_external_docs

    adjust_linked_external_docs(
        "https://numpy.org/doc/stable/", "https://numpy.org/doc/{numpy.__version__}/", _PATH_ROOT
    )
    adjust_linked_external_docs(
        "https://pytorch.org/docs/stable/", "https://pytorch.org/docs/{torch.__version__}/", _PATH_ROOT
    )
    adjust_linked_external_docs(
        "https://lightning.ai/docs/torchmetrics", "https://lightning.ai/docs/torchmetrics/v{torchmetrics.__version__}/", _PATH_ROOT, version_digits=3
    )
    adjust_linked_external_docs(
        "https://lightning.ai/docs/fabric/stable/", "https://lightning.ai/docs/fabric/{lightning_fabric.__version__}/", _PATH_ROOT, version_digits=3
    )
    adjust_linked_external_docs(
        "https://tensorboardx.readthedocs.io/en/stable/", "https://tensorboardx.readthedocs.io/en/v{tensorboard.__version__}/", _PATH_ROOT
    )

# -- Project information -----------------------------------------------------

project = "PyTorch Lightning"
copyright = lightning.__copyright__
author = lightning.__author__

# The short X.Y version
version = lightning.__version__
# The full version, including alpha/beta/rc tags
release = lightning.__version__

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
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    # 'sphinxcontrib.mockautodoc',  # raises error: directive 'automodule' is already registered ...
    # 'sphinxcontrib.fulltoc',  # breaks pytorch-theme with unexpected kw argument 'titles_only'
    "sphinxcontrib.video",
    "myst_parser",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_paramlinks",
    "sphinx_togglebutton",
    "lai_sphinx_theme.extensions.lightning",
    'sphinx.ext.mathjax',
]

# Suppress warnings about duplicate labels (needed for PL tutorials)
suppress_warnings = [
    "autosectionlabel.*",
]

copybutton_prompt_text = ">>> "
copybutton_prompt_text1 = "... "
copybutton_exclude = ".linenos"

copybutton_only_copy_prompt_lines = True

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
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html?highlight=anchor#auto-generated-header-anchors
myst_heading_anchors = 3

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
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    f"{_FOLDER_GENERATED}/PULL_REQUEST_TEMPLATE.md",
    "notebooks/sample-template*",
]

if not _COPY_NOTEBOOKS:
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
html_theme = "lai_sphinx_theme"
html_theme_path = [os.environ.get("LIT_SPHINX_PATH", lai_sphinx_theme.get_html_theme_path())]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "pytorch_project": "https://lightning.ai",
    "canonical_url": lightning.__docs_url__,
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

# MathJax configuration
mathjax3_config = {
    'tex': {
        'packages': {'[+]': ['ams', 'newcommand', 'configMacros']}
    },
}

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
    "torchmetrics": ("https://lightning.ai/docs/torchmetrics/stable/", None),
    "lightning_habana": ("https://lightning-ai.github.io/lightning-Habana/", None),
    "tensorboardX": ("https://tensorboardx.readthedocs.io/en/stable/", None),
    # needed for referencing Fabric from lightning scope
    "lightning.fabric": ("https://lightning.ai/docs/fabric/stable/", None),
    # TODO: these are missing objects.inv
    # "comet_ml": ("https://www.comet.com/docs/v2/", None),
    # "neptune": ("https://docs.neptune.ai/", None),
    # "wandb": ("https://docs.wandb.ai//", None),
}
nitpicky = True


nitpick_ignore = [
    ("py:class", "typing.Self"),
    # missing in generated API
    ("py:exc", "MisconfigurationException"),
    # TODO: generated list of all existing ATM, need to be fixed
    ("py:class", "AveragedModel"),
    ("py:class", "CometExperiment"),
    ("py:meth", "DataModule.__init__"),
    ("py:class", "HPUAccelerator"),
    ("py:class", "Tensor"),
    ("py:class", "_PATH"),
    ("py:func", "add_argument"),
    ("py:func", "add_class_arguments"),
    ("py:meth", "apply_to_collection"),
    ("py:attr", "best_model_path"),
    ("py:attr", "best_model_score"),
    ("py:attr", "checkpoint_path"),
    ("py:class", "comet_ml.ExistingExperiment"),
    ("py:class", "comet_ml.Experiment"),
    ("py:class", "comet_ml.OfflineExperiment"),
    ("py:meth", "deepspeed.DeepSpeedEngine.backward"),
    ("py:attr", "example_input_array"),
    ("py:class", "jsonargparse._core.ArgumentParser"),
    ("py:class", "jsonargparse._namespace.Namespace"),
    ("py:class", "jsonargparse.core.ArgumentParser"),
    ("py:class", "jsonargparse.namespace.Namespace"),
    ("py:class", "transformer_engine.common.recipe.DelayedScaling"),
    ("py:class", "lightning.fabric.accelerators.xla.XLAAccelerator"),
    ("py:class", "lightning.fabric.loggers.csv_logs._ExperimentWriter"),
    ("py:class", "lightning.fabric.loggers.logger._DummyExperiment"),
    ("py:class", "lightning.fabric.plugins.precision.transformer_engine.TransformerEnginePrecision"),
    ("py:class", "lightning.fabric.plugins.precision.bitsandbytes.BitsandbytesPrecision"),
    ("py:class", "lightning.fabric.utilities.data.AttributeDict"),
    ("py:class", "lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin"),
    ("py:func", "lightning.fabric.utilities.seed.seed_everything"),
    ("py:class", "lightning.fabric.utilities.types.LRScheduler"),
    ("py:class", "lightning.fabric.utilities.types.ReduceLROnPlateau"),
    ("py:class", "lightning.fabric.utilities.types.Steppable"),
    ("py:class", "lightning.fabric.wrappers._FabricOptimizer"),
    ("py:class", "lightning.fabric.utilities.throughput.Throughput"),
    ("py:func", "lightning.fabric.utilities.throughput.measure_flops"),
    ("py:class", "lightning.fabric.utilities.spike.SpikeDetection"),
    ("py:meth", "lightning.pytorch.Callback.on_exception"),
    ("py:class", "lightning.pytorch.LightningModule"),
    ("py:meth", "lightning.pytorch.LightningModule.on_train_epoch_end"),
    ("py:meth", "lightning.pytorch.LightningModule.on_validation_epoch_end"),
    ("py:meth", "lightning.pytorch.LightningModule.save_hyperparameters"),
    ("py:meth", "lightning.pytorch.LightningModule.test_step"),
    ("py:meth", "lightning.pytorch.LightningModule.training_step"),
    ("py:meth", "lightning.pytorch.LightningModule.validation_step"),
    ("py:obj", "lightning.pytorch.accelerators.MPSAccelerator"),
    ("py:meth", "lightning.pytorch.accelerators.accelerator.Accelerator.register_accelerators"),
    ("py:paramref", "lightning.pytorch.callbacks.Checkpoint._sphinx_paramlinks_save_top_k"),
    ("py:func", "lightning.pytorch.callbacks.RichProgressBar.configure_columns"),
    ("py:meth", "lightning.pytorch.callbacks.callback.Callback.on_load_checkpoint"),
    ("py:meth", "lightning.pytorch.callbacks.callback.Callback.on_save_checkpoint"),
    ("py:class", "lightning.pytorch.callbacks.checkpoint.Checkpoint"),
    ("py:meth", "lightning.pytorch.callbacks.progress.progress_bar.ProgressBar.get_metrics"),
    ("py:class", "lightning.pytorch.callbacks.progress.rich_progress.RichProgressBarTheme"),
    ("py:class", "lightning.pytorch.callbacks.progress.tqdm_progress.Tqdm"),
    ("py:class", "lightning.pytorch.cli.ReduceLROnPlateau"),
    ("py:meth", "lightning.pytorch.core.LightningDataModule.setup"),
    ("py:meth", "lightning.pytorch.core.LightningModule.configure_model"),
    ("py:meth", "lightning.pytorch.core.LightningModule.save_hyperparameters"),
    ("py:meth", "lightning.pytorch.core.LightningModule.setup"),
    ("py:meth", "lightning.pytorch.core.hooks.ModelHooks.on_after_batch_transfer"),
    ("py:meth", "lightning.pytorch.core.hooks.ModelHooks.setup"),
    ("py:meth", "lightning.pytorch.core.hooks.ModelHooks.transfer_batch_to_device"),
    ("py:meth", "lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin.save_hyperparameters"),
    ("py:class", "lightning.pytorch.loggers.Logger"),
    ("py:func", "lightning.pytorch.loggers.logger.rank_zero_experiment"),
    ("py:class", "lightning.pytorch.plugins.environments.cluster_environment.ClusterEnvironment"),
    ("py:class", "lightning.pytorch.plugins.environments.slurm_environment.SLURMEnvironment"),
    ("py:class", "lightning.pytorch.plugins.io.wrapper._WrappingCheckpointIO"),
    ("py:func", "lightning.pytorch.seed_everything"),
    ("py:class", "lightning.pytorch.serve.servable_module.ServableModule"),
    ("py:class", "lightning.pytorch.serve.servable_module_validator.ServableModuleValidator"),
    ("py:mod", "lightning.pytorch.strategies"),
    ("py:class", "lightning.pytorch.strategies.SingleXLAStrategy"),
    ("py:meth", "lightning.pytorch.strategies.ddp.DDPStrategy.configure_ddp"),
    ("py:meth", "lightning.pytorch.strategies.ddp.DDPStrategy.setup_distributed"),
    ("py:meth", "lightning.pytorch.trainer.trainer.Trainer.lightning_module"),
    ("py:class", "lightning.pytorch.tuner.lr_finder._LRFinder"),
    ("py:class", "lightning.pytorch.utilities.CombinedLoader"),
    ("py:obj", "lightning.pytorch.utilities.deepspeed.ds_checkpoint_dir"),
    ("py:obj", "lightning.pytorch.utilities.memory.is_cuda_out_of_memory"),
    ("py:obj", "lightning.pytorch.utilities.memory.is_cudnn_snafu"),
    ("py:obj", "lightning.pytorch.utilities.memory.is_oom_error"),
    ("py:obj", "lightning.pytorch.utilities.memory.is_out_of_cpu_memory"),
    ("py:func", "lightning.pytorch.utilities.rank_zero.rank_zero_only"),
    ("py:class", "lightning.pytorch.utilities.types.LRSchedulerConfig"),
    ("py:class", "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig"),
    ("py:class", "lightning_habana.pytorch.plugins.precision.HPUPrecisionPlugin"),
    ("py:class", "lightning_habana.pytorch.strategies.HPUParallelStrategy"),
    ("py:class", "lightning_habana.pytorch.strategies.SingleHPUStrategy"),
    ("py:obj", "logger.experiment"),
    ("py:class", "mlflow.tracking.MlflowClient"),
    ("py:attr", "model"),
    ("py:meth", "move_data_to_device"),
    ("py:class", "neptune.Run"),
    ("py:class", "neptune.handler.Handler"),
    ("py:meth", "on_after_batch_transfer"),
    ("py:meth", "on_before_batch_transfer"),
    ("py:meth", "on_save_checkpoint"),
    ("py:meth", "optimizer_step"),
    ("py:class", "out_dict"),
    ("py:meth", "prepare_data"),
    ("py:class", "lightning.pytorch.callbacks.device_stats_monitor.DeviceStatsMonitor"),
    ("py:meth", "setup"),
    ("py:meth", "test_step"),
    ("py:meth", "toggle_optimizer"),
    ("py:class", "torch.ScriptModule"),
    ("py:class", "torch.distributed.fsdp.fully_sharded_data_parallel.CPUOffload"),
    ("py:class", "torch.distributed.fsdp.fully_sharded_data_parallel.MixedPrecision"),
    ("py:class", "torch.distributed.fsdp.fully_sharded_data_parallel.ShardingStrategy"),
    ("py:class", "torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler"),
    ("py:class", "torch.distributed.fsdp.wrap.ModuleWrapPolicy"),
    ("py:func", "torch.inference_mode"),
    ("py:meth", "torch.mean"),
    ("py:func", "torch.nn.Module.eval"),
    ("py:func", "torch.no_grad"),
    ("py:class", "torch.optim.lr_scheduler.LRScheduler"),
    ("py:meth", "torch.set_default_tensor_type"),
    ("py:class", "torch.utils.data.DistributedSampler"),
    ("py:class", "torch_xla.distributed.parallel_loader.MpDeviceLoader"),
    ("py:func", "torch_xla.distributed.xla_multiprocessing.spawn"),
    ("py:class", "torch._dynamo.OptimizedModule"),
    ("py:mod", "tqdm"),
    ("py:meth", "training_step"),
    ("py:meth", "transfer_batch_to_device"),
    ("py:class", "types.FrameType"),
    ("py:class", "typing.TypeGuard"),
    ("py:meth", "untoggle_optimizer"),
    ("py:meth", "validation_step"),
    ("py:class", "wandb.Artifact"),
    ("py:func", "wandb.init"),
    ("py:class", "wandb.sdk.lib.RunDisabled"),
    ("py:class", "wandb.wandb_run.Run"),
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
    "hydra-core": "hydra",
}
MOCK_PACKAGES = []
if _SPHINX_MOCK_REQUIREMENTS:
    _path_require = lambda fname: os.path.join(_PATH_ROOT, "requirements", "pytorch", fname)
    # mock also base packages when we are on RTD since we don't install them there
    MOCK_PACKAGES += package_list_from_file(_path_require("base.txt"))
    MOCK_PACKAGES += package_list_from_file(_path_require("extra.txt"))
    MOCK_PACKAGES += package_list_from_file(_path_require("strategies.txt"))
    MOCK_PACKAGES += package_list_from_file(_path_require("loggers.info"))
    MOCK_PACKAGES += ["comet_ml", "torch_xla", "transformer_engine", "bitsandbytes"]
    MOCK_PACKAGES.remove("jsonargparse")
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
import lightning.pytorch as pl
from torch import nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import _JSONARGPARSE_SIGNATURES_AVAILABLE as _JSONARGPARSE_AVAILABLE
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
from lightning.fabric.loggers.tensorboard import _TENSORBOARD_AVAILABLE, _TENSORBOARDX_AVAILABLE
from lightning.pytorch.loggers.neptune import _NEPTUNE_AVAILABLE
from lightning.pytorch.loggers.comet import _COMET_AVAILABLE
from lightning.pytorch.loggers.mlflow import _MLFLOW_AVAILABLE
from lightning.pytorch.loggers.wandb import _WANDB_AVAILABLE
"""
coverage_skip_undoc_in_source = True

# skip false positive linkcheck errors from anchors
linkcheck_anchors = False

# A timeout value, in seconds, for the linkcheck builder.
linkcheck_timeout = 60

# ignore all links in any CHANGELOG file
linkcheck_exclude_documents = [r"^(.*\/)*CHANGELOG.*$"]

# ignore the following relative links (false positive errors during linkcheck)
linkcheck_ignore = [
    r"installation.html$",
    r"starter/installation.html$",
    r"^../common/trainer.html#trainer-flags$",
    "https://deepgenerativemodels.github.io/assets/slides/cs236_lecture11.pdf",
    "https://www.intel.com/content/www/us/en/products/docs/processors/what-is-a-gpu.html",
    "https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/",  # noqa: E501
    "https://stackoverflow.com/questions/66640705/how-can-i-install-grpcio-on-an-apple-m1-silicon-laptop",
]
