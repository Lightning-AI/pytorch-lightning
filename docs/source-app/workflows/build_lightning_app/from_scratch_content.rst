
**************
WAIT!
**************
Before you build a Lightning App from scratch, see if you can find an app that is similar to what you need
in the `Lightning App Gallery <https://lightning.ai/apps>`_.

Once you find the Lightning App you want, press "Clone & Run" to see it running on the cloud, then download the code
and change what you want!

----

******************
Build from scratch
******************
If you didn't find a Lightning App similar to the one you need, simply create a file named **app.py** with these contents:

.. code:: python

    import lightning as L


    class WordComponent(L.LightningWork):
        def __init__(self, word):
            super().__init__()
            self.word = word

        def run(self):
            print(self.word)


    class LitApp(L.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.hello = WordComponent("hello")
            self.world = WordComponent("world")

        def run(self):
            print("This is a simple Lightning app, make a better app!")
            self.hello.run()
            self.world.run()


    app = L.LightningApp(LitApp())

----

Run the Lightning App
^^^^^^^^^^^^^^^^^^^^^
Run the Lightning App locally:

.. code:: bash

    lightning run app app.py

Run the Lightning App on the cloud:

.. code:: bash

    lightning run app app.py --cloud

----

*************************************
Build a Lightning App from a template
*************************************
If you didn't find an Lightning App similar to the one you need (in the `Lightning App gallery <https://lightning.ai/apps>`_), another option is to start from a template.
The Lightning CLI can generate a template with built-in testing that can be easily published to the
Lightning App Gallery.

Generate a Lightning App with our template generator:

.. code:: bash

    lightning init app your-app-name

You'll see a print-out like this:

.. code:: bash

    ➜ lightning init app your-app-name

    /Users/Your/Current/dir/your-app-name
    INFO: laying out app template at /Users/Your/Current/dir/your-app-name
    INFO:
        Lightning app template created!
        /Users/Your/Current/dir/your-app-name

    run your app with:
        lightning run app your-app-name/your_app_name/app.py

    run it on the cloud to share with your collaborators:
        lightning run app your-app-name/your_app_name/app.py --cloud

----

Modify the Lightning App template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The command above generates a Lightning App file like this:

.. code:: python

    from your_app_name import ComponentA, ComponentB

    import lightning as L


    class LitApp(L.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.component_a = ComponentA()
            self.component_b = ComponentB()

        def run(self):
            self.component_a.run()
            self.component_b.run()


    app = L.LightningApp(LitApp())

Now you can add your own components as you wish!
