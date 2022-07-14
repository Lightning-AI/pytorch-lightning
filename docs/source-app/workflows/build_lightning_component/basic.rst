###################################
Build a Lightning component (basic)
###################################
**Audience:** Users who want to build a Lightning component.

----

*******************************
Why should I build a component?
*******************************
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
Fork and build
**************
Before you build a Lightning component from scratch, see if you can find a component that is similar to what you need
in the `Lightning component Gallery <https://lightning.ai/components>`_.

Once you find the component you want, download the code and change what you want!

----

****************************
Decide between Flow and Work
****************************

.. raw:: html

    <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/flow_work_choice.png"
        alt="Choosing between LightningFlow and LightningWork"
        style="display: block; margin-left: auto; margin-right: auto; width: 70%; max-width: 600px; padding: 20px 0 40px 0"
    >

There are two types of components in Lightning, LightningFlow and LightningWork.

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

******************
Build from scratch
******************
The first option is if you want to build from scratch

----

Option A: Build a LightningFlow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To implement a LightningFlow, simply subclass ``LightningFlow`` and define the run method:

.. code:: python
    :emphasize-lines: 5

    # app.py
    import lightning_app as la


    class LitFlow(lapp.LightningFlow):
        def run(self):
            for i in range(10):
                print(f"{i}: this kind of code belongs in a LightningFlow")


    app = lapp.LightningApp(LitFlow())

run the app

.. code:: bash

    lightning run app app.py

----

Option B: build a LightningWork
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Only implement a LightningWork if this particular piece of code:

- takes more than 1 second to execute
- or requires its own set of cloud resources
- or both

To implement a LightningWork, simply subclass ``LightningWork`` and define the run method:

.. code:: python
    :emphasize-lines: 6

    # app.py
    from time import sleep
    import lightning_app as la


    class LitWork(lapp.LightningWork):
        def run(self):
            for i in range(100000):
                sleep(2.0)
                print(f"{i} LightningWork: work that is long running or may never end (like a server)")

A LightningWork must always be attached to a LightningFlow and explicitely asked to ``run()``:

.. code:: python
    :emphasize-lines: 13, 16

    from time import sleep
    import lightning_app as la


    class LitWork(lapp.LightningWork):
        def run(self):
            for i in range(100000):
                sleep(2.0)
                print(f"{i} LightningWork: work that is long running or may never end (like a server)")


    class LitFlow(lapp.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_work = LitWork()

        def run(self):
            self.lit_work.run()


    app = lapp.LightningApp(LitFlow())

run the app

.. code:: bash

    lightning run app app.py

----

*********************
Build from a template
*********************
If you'd prefer a component template with built-in testing that can be easily published to the
Lightning component gallery, generate it with our template generator:

.. code:: bash

    lightning init component your-component-name

You'll see a print-out like this:

.. code:: bash

    ➜ lightning init component your-component-name
    INFO: laying out component template at /Users/williamfalcon/Developer/opensource/_/lightning/scratch/hello-world
    INFO:
    ⚡ Lightning component template created! ⚡
    /Users/williamfalcon/Developer/opensource/_/lightning/scratch/hello-world

    ...

----

Modify the component template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The command above generates a component file like this:

.. code:: python

    import lightning_app as la


    class TemplateComponent(lapp.LightningWork):
        def __init__(self) -> None:
            super().__init__()
            self.value = 0

        def run(self):
            self.value += 1
            print("welcome to your work component")
            print("this is running inside a work")

Now you can modify the component as you wish!
