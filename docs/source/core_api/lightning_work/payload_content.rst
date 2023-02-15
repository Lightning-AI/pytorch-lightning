
**************************************
What transferring objects does for you
**************************************

Imagine your application is processing some data using `pandas DaFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ and you want to pass that data to another LightningWork (Work). This is what the **Payload API** is meant for.

----

*************************
Use the Lightning Payload
*************************

The Payload enables non JSON-serializable attribute objects to be part of your Work's state and to be communicated to other Works.

Here is an example:

.. code-block:: python

    import lightning as L
    import pandas as pd


    class SourceWork(L.LightningWork):
        def __init__(self):
            super().__init__()
            self.df = None

        def run(self):
            # do some processing

            df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

            # The object you care about needs to be wrapped into a Payload object.
            self.df = L.storage.Payload(df)

            # You can access the original object from the payload using its value property.
            print("src", self.df.value)
            # src  col1  col2
            # 0     1     3
            # 1     2     4

Once the Payload object is attached to your Work's state, it can be passed to another work using the LightningFlow (Flow) as follows:

.. code-block:: python

    import lightning as L
    import pandas as pd


    class DestinationWork(L.LightningWork):
        def run(self, df: L.storage.Payload):
            # You can access the original object from the payload using its value property.
            print("dst", df.value)
            # dst  col1  col2
            # 0     1     3
            # 1     2     4


    class Flow(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.src = SourceWork()
            self.dst = DestinationWork()

        def run(self):
            self.src.run()
            # The pandas DataFrame created by the ``SourceWork``
            # is passed to the ``DestinationWork``.
            # Internally, Lightning pickles and un-pickle the python object,
            # so you receive a copy of the original object.
            self.dst.run(df=self.src.df)


    app = L.LightningApp(Flow())
