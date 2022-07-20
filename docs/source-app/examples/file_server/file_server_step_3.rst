:orphan:

********************************************
3. Implement a File Server Testing Component
********************************************

Let's dive in on how to implement a testing component for a server.

This component needs to test two things:

* The **/upload_file/** endpoint by creating a file and sending its content to it.

* The **/** endpoint listing files, by validating the that previously uploaded file is present in the response.

.. literalinclude:: ./app.py
    :lines: 161-183
