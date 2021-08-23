.. _governance:

Lightning Governance
####################

This document describes governance processes we follow in developing PyTorch Lightning.

Persons of Interest
*******************

.. _bdfl:

BDFL
----
Role: All final decisions related to Lightning.

- William Falcon (`williamFalcon <https://github.com/williamFalcon>`_) (Lightning founder)


Leads
-----
- Jirka Borovec (`Borda <https://github.com/Borda>`_)
- Ethan Harris (`ethanwharris <https://github.com/ethanwharris>`_) (Torchbearer founder)
- Justus Schock (`justusschock <https://github.com/justusschock>`_) (Former Core Member PyTorch Ignite)
- Adrian Wälchli (`awaelchli <https://github.com/awaelchli>`_)
- Thomas Chaton (`tchaton <https://github.com/tchaton>`_)
- Sean Narenthiran (`SeanNaren <https://github.com/SeanNaren>`_)
- Carlos Mocholí (`carmocca <https://github.com/carmocca>`_)
- Kaushik Bokka (`kaushikb11 <https://github.com/kaushikb11>`_)
- Ananth Subramaniam (`ananthsub <https://github.com/ananthsub>`_)

Core Maintainers
----------------
- Nicki Skafte (`skaftenicki <https://github.com/SkafteNicki>`_)
- Rohit Gupta (`rohitgr7 <https://github.com/rohitgr7>`_)
- Roger Shieh (`s-rog <https://github.com/s-rog>`_)
- Akihiro Nitta (`akihironitta <https://github.com/akihironitta>`_)

Board
-----
- Jeremy Jordan (`jeremyjordan <https://github.com/jeremyjordan>`_)
- Tullie Murrell (`tullie <https://github.com/tullie>`_)
- Nic Eggert (`neggert <https://github.com/neggert>`_)
- Matthew Painter (`MattPainter01 <https://github.com/MattPainter01>`_) (Torchbearer founder)


Alumni
------
- Jeff Yang (`ydcjeff <https://github.com/ydcjeff>`_)
- Jeff Ling (`jeffling <https://github.com/jeffling>`_)
- Teddy Koker (`teddykoker <https://github.com/teddykoker>`_)
- Nate Raw (`nateraw <https://github.com/nateraw>`_)
- Peter Yu (`yukw777 <https://github.com/yukw777>`_)


Releases
********

We release a new minor version (e.g., 1.5.0) every three months and bugfix releases every week.
The minor versions contain new features, API changes, deprecations, removals, potential backward-incompatible
changes and also all previous bugfixes included in any bugfix release. With every release, we publish a changelog
where we list additions, removals, changed functionality and fixes.

Project Management and Decision Making
**************************************

The decision what goes into a release is governed by the :ref:`staff contributors and leaders <governance>` of
Lightning development. Whenever possible, discussion happens publicly on GitHub and includes the whole community.
For controversial changes, it is mandatory to seek consultation from :ref:`BDFL` for a final decision.
When a consensus is reached, staff and core contributors assign milestones and labels to the issue and/or pull request
and start tracking the development. It is possible that priorities change over time.

Commits to the project are exclusively to be added by pull requests on GitHub and anyone in the community is welcome to
review them. However, reviews submitted by
`code owners <https://github.com/PyTorchLightning/pytorch-lightning/blob/master/.github/CODEOWNERS>`_
have higher weight and it is necessary to get the approval of code owners before a pull request can be merged.
Additional requirements may apply case by case.

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

The "X+2" rule is a recommendation and not a strict requirement. Longer deprecation cylces may apply for some cases.

New API and features are declared as

- *Experimental*: Anything labelled as *experimental* or *beta* in the documentation is considered unstable and should
    not be used in production. The community is encouraged to test the feature and report issues directly on GitHub.
- *Stable*: Everything not specifically labelled as experimental should be considered stable. Reported issues will be
    treated with priority.
