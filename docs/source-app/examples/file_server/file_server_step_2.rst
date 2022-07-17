:orphan:

**********************************************************
2. Implement the File Server upload and list_files methods
**********************************************************

Let's dive in on how to implement such methods.

***************************
Implement the upload method
***************************

In this method, we are creating a stream between the uploaded file and the uploaded file stored on the file server disk.

Once the file is uploaded, we are putting the file into the :class:`~lightning_app.storage.drive.Drive`, so it becomes persistent and accessible to all components.

.. literalinclude:: ./app.py
    :lines: 13, 52-100
    :emphasize-lines: 49

*******************************
Implement the fist_files method
*******************************

First, in this method, we get the file in the file server filesystem, if available in the Drive. Once done, we list the the files under the provided paths and return the results.

.. literalinclude:: ./app.py
    :lines: 13, 101-131
    :emphasize-lines: 9


*******************
Implement utilities
*******************

.. literalinclude:: ./app.py
    :lines: 13, 46-51
