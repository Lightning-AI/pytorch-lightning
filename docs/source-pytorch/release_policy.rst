.. _release_policy:

Release Policy
##############

Release Cycle
*************

We release a new minor version (e.g., 1.5.0) every three months and bugfix releases every week.
The minor versions contain new features, API changes, deprecations, removals, potential backward-incompatible
changes and also all previous bugfixes included in any bugfix release. With every release, we publish a changelog
where we list additions, removals, changed functionality and fixes.


API Evolution
*************

Lightning's development is driven by research and best practices in a rapidly developing field of AI and machine
learning. Change is inevitable and when it happens, the Lightning team is committed to minimizing user friction and
maximizing ease of transition from one version to the next. We take backward compatibility and reproducibility very
seriously.

For API removal, renaming or other forms of backward-incompatible changes, the procedure is:

#. A deprecation process is initiated at version X, producing warning messages at runtime and in the documentation.
#. Calls to the deprecated API remain unchanged in their function during the deprecation phase.
#. Two minor versions in the future at version X+2 the breaking change takes effect.

The "X+2" rule is a recommendation and not a strict requirement. Longer deprecation cycles may apply for some cases.

New API and features are declared as:

- *Experimental*: Anything labelled as *experimental* or *beta* in the documentation is considered unstable and should
    not be used in production. The community is encouraged to test the feature and report issues directly on GitHub.
- *Stable*: Everything not specifically labelled as experimental should be considered stable. Reported issues will be
    treated with priority.
