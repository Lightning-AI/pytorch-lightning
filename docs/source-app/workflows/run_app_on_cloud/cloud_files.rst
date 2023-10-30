.. _ignore:

##################################
Configure Your Lightning Cloud App
##################################

**Audience:** Users who want to control Lightning App files on the cloud.

----

**************************************
Ignore file uploads to Lightning cloud
**************************************
Running Lightning Apps on the cloud will upload the source code of your app to the cloud. You can use ``.lightningignore`` file(s) to ignore files or directories while uploading. The `.lightningignore` file follows the same format as a `.gitignore`
file.

For example, the source code directory below with the ``.lightningignore`` file will ignore the file named
``model.pt`` and directory named ``data_dir``.

.. code:: bash

    .
    ├── README.md
    ├── app.py
    ├── data_dir
    │    ├── image1.png
    │    ├── image2.png
    │    └── ...
    ├── .lightningignore
    ├── requirements.txt
    └── model.pt

.. code:: bash

    ~/project/home ❯ cat .lightningignore
    model.pt
    data_dir

A sample ``.lightningignore`` file can be found `here <https://github.com/Lightning-AI/lightning/blob/master/.lightningignore>`_.

If you are a component author and your components creates local files that you want to ignore, you can do:

.. code-block:: python

    class MyComponent(L.LightningWork):  # or L.LightningFlow
        def __init__(self):
            super().__init__()
            self.lightningignore = ("model.pt", "data_dir")


This has the benefit that the files will be ignored automatically for all the component users, making an easier
transition between running locally vs in the cloud.

----

*******************
Structure app files
*******************

We recommend your app contain the following files:

.. code:: bash

    .
    ├── .lightning        (auto-generated- contains Lightning configuration)
    ├── .lightningignore  (contains files not to upload to the cloud)
    ├── app.py
    ├── README.md         (optional- a markdown description of your app)
    └── requirements.txt  (optional- contains all your app dependencies)
