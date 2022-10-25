.. _versioning:

Versioning Policy
#################

PyTorch Lightning follows its own versioning policy but not `semantic versioning (SemVer) <https://semver.org/>`_.

Versioning
**********

A Lightning release number is in the format of ``MAJOR.MINOR.PATCH``.

- A patch release contains only bug fixes. Since it introduces no breaking changes, we recommend users always update the package to the latest version within the minor version whenever possible.
- A minor release, unlike SemVer, contains backwards-incompatible changes, such as API changes and removals, as well as new features, deprecations and all bugfixes since the last release.

With every release, we publish a changelog where we list additions, removals, deprecations, changed functionality and fixes.

API Stability
*************

In Lightning, all API and features are marked as either stable or experimental.

Experimental API
----------------

Experimental APIs are labelled as experimental or beta in the documentation and/or in the release note and are considered unstable and should not be used in production.

For experimental features, any of the following may be true:

- The feature has unstable dependencies.
- The API may change without notice in future versions.
- The performance of the feature has not been verified.
- The docs for this feature are under active development.

Stable API
----------

Everything not specifically labelled as experimental is stable.

For stable APIs, all of the following are true:

- The API is not expected to change.
- If anything does change, we show a deprecation warning before applying the breaking change following the rule described below.

API Evolution
*************

Lightning's development is driven by research and best practices in a rapidly developing field of AI and machine learning. Change is inevitable and when it happens, the Lightning team is committed to minimizing user friction and maximizing ease of transition from one version to the next. We take backwards compatibility and reproducibility very seriously.

For API removal, renaming or other forms of backwards-incompatible changes, the procedure is:

#. A deprecation process is initiated at a minor version ``X``, producing a deprecation warning at runtime and in the documentation.
#. The deprecated API remains unchanged during the deprecation phase for two minor versions.
#. The breaking change takes effect at a minor version ``X+2``.
#. From version ``X+2`` onward, the deprecation warning gets converted into a helpful error, which will remain as long as possible.

The ``X+2`` rule is a recommendation and not a strict requirement. Shorter or longer deprecation cycles may apply to some cases. In the past, DDP2 was removed without a deprecation process because the feature was broken and unusable beyond fixing as discussed in `#12584 <https://github.com/Lightning-AI/lightning/issues/12584>`_. Also, `#10410 <https://github.com/Lightning-AI/lightning/issues/10410>`_ is an example that a longer deprecation applied to. We deprecated the accelerator arguments, such as ``Trainer(gpus=...)``, in 1.7, however, because the APIs were so core that they would impact almost all use cases, we decided not to introduce the breaking change until 2.0.

Python Support
**************

PyTorch Lightning follows `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_ which PyTorch also follows (`#74203 <https://github.com/pytorch/pytorch/issues/74203>`_).

PyTorch Support
***************

PyTorch Lightning supports the latest four minor versions of PyTorch at the time of release. For example, PyTorch Lightning 1.7 supports PyTorch 1.9, 1.10, 1.11 and 1.12.
