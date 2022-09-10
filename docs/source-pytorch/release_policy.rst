.. _release_policy:

Release Policy
##############

This document describes release policy we follow for PyTorch Lightning.

Versioning
**********

A Lightning release number is in the format of ``MAJOR.MINOR.PATCH``.

We release a new minor version every three months and bugfix release every week.

Patch releases contain only bugfixes and introduce no breaking changes, so it is always recommended to update the package to the latest version within the minor version whenever possible.
However, unlike `semantic versioning (SemVer) <https://semver.org/>`_, minor releases contain API changes, removals, potential backward-incompatible
changes as well as new features, deprecations, and all previous bugfixes included in any bugfix release. With every release, we publish a changelog
where we list additions, removals, deprecations, changed functionality and fixes.

API Evolution
*************

Lightning's development is driven by research and best practices in a rapidly developing field of AI and machine
learning. Change is inevitable and when it happens, the Lightning team is committed to minimizing user friction and
maximizing ease of transition from one version to the next. We take backward compatibility and reproducibility very
seriously.

For API removal, renaming or other forms of backward-incompatible changes, the procedure is:

#. A deprecation process is initiated at a minor version ``X``, producing warning messages at runtime and in the documentation.
#. Calls to the deprecated API remain unchanged in their function during the deprecation phase.
#. Two minor versions in the future at version ``X+2`` the breaking change takes effect.

The "X+2" rule is a recommendation and not a strict requirement. Longer deprecation cycles may apply for some cases.

In Lightning, all API and features are marked as:

- *Experimental*: Anything labelled as *experimental* or *beta* in the documentation is considered unstable and should
    not be used in production. The community is encouraged to test the feature and report issues directly on GitHub.
- *Stable*: Everything not specifically labelled as experimental should be considered stable. Reported issues will be
    treated with priority.
