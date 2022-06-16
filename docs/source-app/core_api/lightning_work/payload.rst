:orphan:

#############################
Sharing Objects between Works
#############################

**Audience:** Users who want to know how to transfer python objects between their works.

**Level:** Advanced

**Prerequisite**: Know about the pandas library and read the :ref:`access_app_state` guide.

----

************************************
When do I need to transfer objects ?
************************************

Imagine your application is processing some data using `pandas DaFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ and you want to pass those data to another work. This is when and what the **Payload API** is meant for.


*************************************
How can I use the Lightning Payload ?
*************************************

The Payload enables non JSON-serializable attribute objects to be part of your work state and be communicated to other works.

Here is an example how to use it:

.. code-block:: python

    import lightning_app as la
    import pandas as pd


    class SourceWork(lapp.LightningWork):
        def __init__(self):
            super().__init__()
            self.df = None

        def run(self):
            # do some processing

            df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

            # The object you care about needs to be wrapped into a Payload object.
            self.df = lapp.storage.Payload(df)

            # You can access the original object from the payload using its value property.
            print("src", self.df.value)
            # src  col1  col2
            # 0     1     3
            # 1     2     4

Once the Payload object is attached to your work state, it can be passed to another work via the flow as follows:

.. code-block:: python

    import lightning_app as la
    import pandas as pd


    class DestinationWork(lapp.LightningWork):
        def run(self, df: lapp.storage.Payload):
            # You can access the original object from the payload using its value property.
            print("dst", df.value)
            # dst  col1  col2
            # 0     1     3
            # 1     2     4


    class Flow(lapp.LightningFlow):
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


    app = lapp.LightningApp(Flow())
