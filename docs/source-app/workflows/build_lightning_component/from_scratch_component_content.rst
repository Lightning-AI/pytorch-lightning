*******************************
LightningFlow vs. LightningWork
*******************************

.. _flow_vs_work:

.. raw:: html

    <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/flow_work_choice.png"
        alt="Choosing between LightningFlow and LightningWork"
        style="display: block; margin-left: auto; margin-right: auto; width: 70%; max-width: 600px; padding: 20px 0 40px 0"
    >

There are two types of components in Lightning, **LightningFlow** and **LightningWork**.

Use a **LightningFlow** component for any programming logic that runs in less than 1 second.

.. code:: python

    for i in range(10):
        print(f"{i}: this kind of code belongs in a LightningFlow")

Use a **LightningWork** component for any programming logic that takes more than 1 second or requires its own hardware.

.. code:: python

    from time import sleep

    for i in range(100000):
        sleep(2.0)
        print(f"{i} LightningWork: work that is long running or may never end (like a server)")

----

**************************************************
What developing a Lightning Component does for you
**************************************************
Lightning Components break up complex systems into modular components. The first obvious benefit is that components
can be reused across other apps. This means you can build once, test it and forget it.

As a researcher it also means that your code can be taken to production without needing a team of engineers to help
productionize it.

As a machine learning engineer, it means that your cloud system is:

- fault tolerant
- cloud agnostic
- testable (unlike YAML/CI/CD code)
- version controlled
- enables cross-functional collaboration

----

**************
WAIT!
**************
Before you build a Lightning component from scratch, see if you can find a component that is similar to what you need
in the `Lightning component Gallery <https://lightning.ai/components>`_.

Once you find the component you want, download the code and change what you want!

----

*****************************************
Build a Lighitning component from scratch
*****************************************
If you didn't find a Lightning component similar to the one you need, you can build one from scratch.

----

Build a LightningFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To implement a LightningFlow, simply subclass ``LightningFlow`` and define the run method:

.. code:: python
    :emphasize-lines: 5

    # app.py
    import lightning as L


    class LitFlow(L.LightningFlow):
        def run(self):
            for i in range(10):
                print(f"{i}: this kind of code belongs in a LightningFlow")


    app = L.LightningApp(LitFlow())

run the app

.. code:: bash

    lightning_app run app app.py

----

Build a LightningWork
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Only implement a LightningWork if this particular piece of code:

- takes more than 1 second to execute
- requires its own set of cloud resources
- or both

To implement a LightningWork, simply subclass ``LightningWork`` and define the run method:

.. code:: python
    :emphasize-lines: 6

    # app.py
    from time import sleep
    import lightning as L


    class LitWork(L.LightningWork):
        def run(self):
            for i in range(100000):
                sleep(2.0)
                print(f"{i} LightningWork: work that is long running or may never end (like a server)")

A LightningWork must always be attached to a LightningFlow and explicitly asked to ``run()``:

.. code:: python
    :emphasize-lines: 13, 16

    from time import sleep
    import lightning as L


    class LitWork(L.LightningWork):
        def run(self):
            for i in range(100000):
                sleep(2.0)
                print(f"{i} LightningWork: work that is long running or may never end (like a server)")


    class LitFlow(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_work = LitWork()

        def run(self):
            self.lit_work.run()


    app = L.LightningApp(LitFlow())

run the app

.. code:: bash

    lightning_app run app app.py
