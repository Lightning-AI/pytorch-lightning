:orphan:

###################
Level 9: Event loop
###################
**Audience:** Users who want to build reactive Lightning Apps and move beyond DAGs.

**Prereqs:** Level 8+

----

Drawing inspiration from modern web frameworks like `React.js <https://reactjs.org/>`_, the Lightning App runs all flows in an **event loop** (forever), which is triggered several times a second after collecting any works' state change.

.. figure::  https://pl-public-data.s3.amazonaws.com/assets_lightning/lightning_loop.gif

When running a Lightning App in the cloud, the ``LightningWork`` run on different machines. LightningWork communicates any state changes to the **event loop** which re-executes the flow with the newly-collected works' state.
