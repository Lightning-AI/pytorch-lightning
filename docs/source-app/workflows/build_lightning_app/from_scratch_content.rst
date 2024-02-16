
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

    lightning_app run app app.py

Run the Lightning App on the cloud:

.. code:: bash

    lightning_app run app app.py --cloud
