.. _versioning:

Versioning Policy
#################

PyTorch Lightning follows its own versioning policy which differs from `semantic versioning (SemVer) <https://semver.org/>`_.

Versioning
**********

A Lightning release number is in the format of ``MAJOR.MINOR.PATCH``.

- A patch release contains only bug fixes. Since it introduces no breaking changes, we recommend users always update the package to the latest version within the minor version whenever possible.
- A minor release may contain backwards-incompatible changes **with deprecations** (unlike SemVer), such as API changes and removals, as well as new features and bugfixes since last release.
- A major release may contain backwards-incompatible changes **without deprecations**, as well as new features, and bugfixes since last release.

With every release, we publish a changelog where we list additions, removals, deprecations, changed functionality and fixes.

The ``lightning.app`` package is an exception to this rule, as it may contain any change with or without deprecations in any of the releases.

API Stability
*************

In Lightning, all public APIs are considered stable unless explicitly marked as experimental in their documentation or docstrings.
Modules, functions, classes, and methods that are protected (have a leading underscore, see https://peps.python.org/pep-0008/ for more information) may be changed or removed at any time.

Stable API
----------

Everything not specifically labelled as experimental is stable.

For stable APIs, all of the following are true:

- The API is not expected to change.
- If anything does change, we show a deprecation warning before applying the breaking change following the policy described in the "API Evolution" section below.

Experimental API
----------------

Experimental APIs are labelled as experimental in their documentation or docstrings.
For experimental features, any of the following may be true:

- The feature uses dependencies that are under active development and may change outside our control.
- The API may change without notice in future versions.
- The performance of the feature has not been verified.
- The feature has not been battle tested by the core team in production scenarios.
- The feature is under active development.

While we may still issue deprecation warnings for experimental API changes, this is not guaranteed.
Therefore, it is important to be cautious when using experimental features and be prepared to modify your code if the
API changes in a future release. In this case, you might want to pin your dependencies to avoid unexpected issues.

API Evolution
*************

Lightning's development is driven by research and best practices in a rapidly developing field of AI and machine learning. Change is inevitable and when it happens, the Lightning team is committed to minimizing user friction and maximizing ease of transition from one version to the next. We take backwards compatibility and reproducibility very seriously.

For API removal, renaming or other forms of backwards-incompatible changes, the procedure is:

#. A deprecation process is initiated at a minor version ``MAJOR.MINOR.PATCH`` (e.g. ``1.5.0``), producing a deprecation warning at runtime and removing it from the documentation.
#. The deprecated API remains unchanged during the deprecation phase for two minor versions or the next major update, whichever comes first.
#. The breaking change is done in version ``MAJOR.(MINOR+2).0`` (e.g. ``1.7.0``), or ``(MAJOR+1).0.0`` (e.g. ``2.0.0``), whichever comes first.
#. From that version onward, the deprecation warning gets converted into a helpful error, which will remain until next major release.

This policy is not strict. Shorter or longer deprecation cycles may apply to some cases.
For example, in the past DDP2 was removed without a deprecation process because the feature was broken and unusable beyond fixing as discussed in `#12584 <https://github.com/Lightning-AI/lightning/issues/12584>`_.
Also, `#10410 <https://github.com/Lightning-AI/lightning/issues/10410>`_ is an example that a longer deprecation applied to. We deprecated the accelerator arguments, such as ``Trainer(gpus=...)``, in 1.7, however, because the APIs were so core that they would impact almost all use cases, we decided not to introduce the breaking change until 2.0.

Compatibility matrix
********************

PyTorch Lightning follows `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_ which PyTorch also follows (`#74203 <https://github.com/pytorch/pytorch/issues/74203>`_).
The table below indicates the coverage of tested versions in our CI. Versions outside the ranges may unofficially work in some cases.

.. list-table::
   :header-rows: 1

   * - ``lightning.pytorch``
     - ``pytorch_lightning``
     - ``lightning.fabric``
     - ``torch``
     - ``torchmetrics``
     - Python
   * - 2.2
     - 2.2
     - 2.2
     - ≥1.13, ≤2.2
     - ≥0.7.0
     - ≥3.8, ≤3.11
   * - 2.1
     - 2.1
     - 2.1
     - ≥1.12, ≤2.1
     - ≥0.7.0
     - ≥3.8, ≤3.11
   * - 2.0
     - 2.0
     - 2.0 (GA)
     - ≥1.11, ≤2.0
     - ≥0.7.0
     - ≥3.8, ≤3.10
   * - 1.9
     - 1.9
     - 1.9 (experimental)
     - ≥1.10, ≤1.13
     - ≥0.7.0
     - ≥3.7, ≤3.10
   * - 1.8**
     - 1.8
     - n/a***
     - ≥1.10, ≤1.13
     - ≥0.7.0
     - ≥3.7, ≤3.10
   * - n/a
     - 1.7
     - n/a***
     - ≥1.9, ≤1.12
     - ≥0.7.0
     - ≥3.7, ≤3.10
   * - n/a
     - 1.6
     - n/a***
     - ≥1.8, ≤1.11
     - ≥0.4.1
     - ≥3.7, ≤3.9
   * - n/a
     - 1.5
     - n/a***
     - ≥1.7, ≤1.10
     - ≥0.4.1
     - ≥3.6, ≤3.9
   * - n/a
     - 1.4
     - n/a
     - ≥1.6, ≤1.9
     - ≥0.4.0
     - ≥3.6, ≤3.9
   * - n/a
     - 1.3
     - n/a
     - ≥1.4, ≤1.8
     - ≥0.2.0
     - ≥3.6, ≤3.9
   * - n/a
     - 1.2
     - n/a
     - ≥1.4, ≤1.8
     - n/a*
     - ≥3.6, ≤3.8
   * - n/a
     - 1.1
     - n/a
     - ≥1.3, ≤1.8
     - n/a*
     - ≥3.6, ≤3.8
   * - n/a
     - 1.0
     - n/a
     - ≥1.3, ≤1.7
     - n/a*
     - ≥3.6, ≤3.8

\* ``torchmetrics`` was part of ``pytorch_lightning`` at the time and was decoupled to a separate package in v1.3.

\*\* The joint ``lightning`` package was first published in version 1.8

\*\*\* Fabric is the evolution of ``LightningLite`` which was released inside ``pytorch_lightning`` 1.5 and was decoupled to a separate package in v1.9
