:orphan:

#####################################
Start from Ready-to-Run Template Apps
#####################################

.. _jumpstart_from_app_gallery:

Anyone can build Apps for their own use cases and promote them on the `App Gallery <https://lightning.ai/apps>`_.

In return, you can benefit from the work of others and get started faster by re-using a ready-to-run App close to your own use case.


*************
User Workflow
*************

#. Visit the `App Gallery <https://lightning.ai/apps>`_ and look for an App close to your own use case.

    .. raw:: html

       <br />

#. If **Launch** is available, it means the App is live and ready to be used! Take it for a spin.

    .. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/launch_button.png
        :alt: Launch Button on lightning.ai
        :width: 100 %

#. By clicking **Clone & Run**, a copy of the App is added to your account and an instance starts running.


    .. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/clone_and_run.mp4
        :poster: https://pl-public-data.s3.amazonaws.com/assets_lightning/clone_and_run.png
        :width: 600
        :class: background-video
        :autoplay:
        :loop:
        :muted:

#. If you found an App that matches what you need, move to **step 5**! Otherwise, go back to **step 1**.

    .. raw:: html

       <br />

#. Copy the installation command (optionally from the clipboard on the right).

    .. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/install_command.png
        :alt: Install command on lightning.ai
        :width: 100 %

#. Copy the command to your local terminal.

    .. code-block:: bash

        lightning_app install app lightning/hackernews-app

#. Go through the installation steps.

    .. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/install_an_app.mp4
        :poster: https://pl-public-data.s3.amazonaws.com/assets_lightning/install_an_app.png
        :width: 600
        :class: background-video
        :autoplay:
        :loop:
        :muted:

#. Run the App locally.

    .. code-block:: bash

        cd LAI-Hackernews-App
        lightning_app run app app.py

    .. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/hackernews.mp4
        :poster: https://pl-public-data.s3.amazonaws.com/assets_lightning/hackernews.png
        :width: 600
        :class: background-video
        :autoplay:
        :loop:
        :muted:

#. Open the code with your favorite IDE, modify it, and run it back in the cloud.

    .. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/hackernews_modified.mp4
        :poster: https://pl-public-data.s3.amazonaws.com/assets_lightning/hackernews_modified.png
        :width: 600
        :class: background-video
        :autoplay:
        :loop:
        :muted:

----

**********
Next Steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Add Component made by others to your App
   :description: Add more functionality to your projects
   :col_css: col-md-6
   :button_link: jumpstart_from_component_gallery.html
   :height: 180

.. displayitem::
   :header: Level-up your skills with Lightning Apps
   :description: From Basic to Advanced Skills
   :col_css: col-md-6
   :button_link: ../levels/basic/index.html
   :height: 180

.. raw:: html

      </div>
   </div>
   <br />
