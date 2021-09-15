.. _loop_customization:

Loop Customization
==================

Loop customization is an experimental feature introduced in Lightning 1.5 that enables advanced users to write new training logic or modify existing behavior in Lightning's training-, evaluation-, or prediction loops.
By advanced users we mean users that are familiar with the major components under the Trainer and how they interact with the LightningModule.

In this advanced user guide we will learn about

- how the Trainer runs a loop,
- the Loop base class,
- the default loop implementations and subloops Lightning has,
- how Lightning defines a tree structure of loops and subloops,
- how you can create a custom loop for a new training step flavor,
- and how you can connect the custom loop and run it.

Most importantly, we will also provide guidelines when to use loop customization and when NOT to use it.


Trainer entry points for loops
------------------------------
