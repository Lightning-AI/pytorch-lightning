.. _governance:

Lightning Governance
####################

This document describes governance processes we follow in developing PyTorch Lightning.

Persons of Interest
*******************

.. _governance_bdfl:

BDFL
----
Role: All final decisions related to Lightning.

- William Falcon (`williamFalcon <https://github.com/williamFalcon>`_) (Lightning founder)


Leads
-----
- Jirka Borovec (`Borda <https://github.com/Borda>`_)
- Ethan Harris (`ethanwharris <https://github.com/ethanwharris>`_) (Torchbearer founder)
- Justus Schock (`justusschock <https://github.com/justusschock>`_)
- Adrian Wälchli (`awaelchli <https://github.com/awaelchli>`_)
- Thomas Chaton (`tchaton <https://github.com/tchaton>`_)
- Sean Narenthiran (`SeanNaren <https://github.com/SeanNaren>`_)
- Carlos Mocholí (`carmocca <https://github.com/carmocca>`_)
- Kaushik Bokka (`kaushikb11 <https://github.com/kaushikb11>`_)
- Ananth Subramaniam (`ananthsub <https://github.com/ananthsub>`_)
- Rohit Gupta (`rohitgr7 <https://github.com/rohitgr7>`_)


Core Maintainers
----------------
- Nicki Skafte (`skaftenicki <https://github.com/SkafteNicki>`_)
- Akihiro Nitta (`akihironitta <https://github.com/akihironitta>`_)
- Danielle Pintz (`daniellepintz <https://github.com/daniellepintz>`_)
- Siyu Wang (`four4fish <https://github.com/four4fish>`_)


Board
-----
- Jeremy Jordan (`jeremyjordan <https://github.com/jeremyjordan>`_)
- Tullie Murrell (`tullie <https://github.com/tullie>`_)
- Nic Eggert (`neggert <https://github.com/neggert>`_)
- Matthew Painter (`MattPainter01 <https://github.com/MattPainter01>`_) (Torchbearer founder)


Alumni
------
- Jeff Yang (`ydcjeff <https://github.com/ydcjeff>`_)
- Lezwon Castelino (`lezwon <https://github.com/lezwon>`_)
- Jeff Ling (`jeffling <https://github.com/jeffling>`_)
- Teddy Koker (`teddykoker <https://github.com/teddykoker>`_)
- Nate Raw (`nateraw <https://github.com/nateraw>`_)
- Peter Yu (`yukw777 <https://github.com/yukw777>`_)
- Roger Shieh (`s-rog <https://github.com/s-rog>`_)

Project Management and Decision Making
**************************************

The decision what goes into a release is governed by the :ref:`staff contributors and leaders <governance>` of
Lightning development. Whenever possible, discussion happens publicly on GitHub and includes the whole community.
For controversial changes, it is mandatory to seek consultation from :ref:`governance_bdfl` for a final decision.
When a consensus is reached, staff and core contributors assign milestones and labels to the issue and/or pull request
and start tracking the development. It is possible that priorities change over time.

Commits to the project are exclusively to be added by pull requests on GitHub and anyone in the community is welcome to
review them. However, reviews submitted by
`code owners <https://github.com/Lightning-AI/lightning/blob/master/.github/CODEOWNERS>`_
have higher weight and it is necessary to get the approval of code owners before a pull request can be merged.
Additional requirements may apply case by case.
