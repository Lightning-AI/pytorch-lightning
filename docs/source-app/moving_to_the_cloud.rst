:orphan:

.. _moving_to_the_cloud:

####################
Moving to the Cloud
####################

.. warning:: This is in progress and not yet fully supported.

In the :ref:`quick_start` guide, you learned how to implement a simple app
that trains an image classifier and serve it once trained.

In this tutorial, you'll learn how to extend that application so that it works seamlessly
both locally and in the cloud.

----

********************************
Step 1: Distributed Application
********************************


Distributed Storage
^^^^^^^^^^^^^^^^^^^

When running your application in a fully-distributed setting, the data available on one machine won't necessarily be available on another.

To solve this problem, Lightning introduces the :class:`~lightning.app.storage.path.Path` object.
This ensures that your code can run both locally and in the cloud.

The :class:`~lightning.app.storage.path.Path` object keeps track of the work which creates
the path. This enables Lightning to transfer the files correctly in a distributed setting.

Instead of passing a string representing a file or directory, Lightning simply wraps
them into a :class:`~lightning.app.storage.path.Path` object and makes them an attribute of your LightningWork.

Without doing this conscientiously for every single path, your application will fail in the cloud.

In the example below, a file written by **SourceFileWork** is being transferred by the flow
to the **DestinationFileAndServeWork** work. The Path object is the reference to the file.

.. literalinclude:: ../../examples/app/boring/app.py
    :emphasize-lines: 5, 22, 28, 48

In the ``scripts/serve.py`` file, we are creating a **FastApi Service** running on port ``1111``
that returns the content of the file received from **SourceFileWork** when
a post request is sent to ``/file``.

.. literalinclude:: ../../examples/app/boring/scripts/serve.py
    :emphasize-lines: 21, 23-26

----

Distributed Frontend
^^^^^^^^^^^^^^^^^^^^

In the above example, the **FastAPI Service** was running on one machine,
and the frontend UI in another.

In order to assemble them, you need to do two things:

* Provide **port** argument to your work's ``__init__`` method to expose a single service.

Here's how to expose the port:

.. literalinclude:: ../../examples/app/boring/app.py
    :emphasize-lines: 8
    :lines: 33-44


And here's how to expose your services within the ``configure_layout`` flow hook:

.. literalinclude:: ../../examples/app/boring/app.py
    :emphasize-lines: 5
    :lines: 53-57

In this example, we're appending ``/file`` to our **FastApi Service** url.
This means that our ``Boring Tab`` triggers the ``get_file_content`` from the **FastAPI Service**
and embeds its content as an `IFrame <https://en.wikipedia.org/wiki/HTML_element#Frames>`_.

.. literalinclude:: ../../examples/app/boring/scripts/serve.py
    :lines: 23-26


Here's a visualization of the application described above:

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/storage_ui.gif
    :alt: Storage API Animation
    :width: 100 %

----

*****************************
Step 2: Scalable Application
*****************************

The benefit of defining long-running code inside a
:class:`~lightning.app.core.work.LightningWork`
component is that you can run it on different hardware
by providing :class:`~lightning.app.utilities.packaging.cloud_compute.CloudCompute` to
the ``__init__`` method of your :class:`~lightning.app.core.work.LightningWork`.

By adapting the :ref:`quick_start` example as follows, you can easily run your component on multiple GPUs:


Without doing much, you’re now running a script on its own cluster of machines! 🤯

----

*****************************
Step 3: Resilient Application
*****************************

We designed Lightning with a strong emphasis on supporting failure cases.
The framework shines when the developer embraces our fault-tolerance best practices,
enabling them to create ML applications with a high degree of complexity as well as a strong support
for unhappy cases.

An entire section would be dedicated to this concept.

TODO
