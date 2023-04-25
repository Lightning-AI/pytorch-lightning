:orphan:

##################################
Differences between Drive and Path
##################################

**Audience:** Users who want to share files between components.


The Lightning Storage system makes it easy to share files between LightningWork so you can run your app both locally and in the cloud without changing the code.


Lightning storage provides two solutions :class:`~lightning.app.storage.drive.Drive` and :class:`~lightning.app.storage.path.Path` to deal with files locally and in the cloud likewise.


----

*****************
What is a Drive ?
*****************

The Drive object provides a central place for your components to share data.

The drive acts as an isolate folder and any component can access it by knowing its name.

Your components can put, list, get, delete files from and to the Drive (except LightningFlow's).

----

****************
What is a Path ?
****************

The Path object is a reference to a specific file or directory from a LightningWork and can be used to transfer those files to another LightningWork (one way, from source to destination).

A good mental representation of the Path Object usage is the `relay race <https://en.wikipedia.org/wiki/Relay_race>`_.
To make a transfer, the LightningWork Receiver asks (e.g when the path object is passed by the flow to the Receiver)
for a copy of the files (baton) owned by their LightningWork Producer (e.g the work that created the files).

----

*********************************
When should I use Drive vs Path ?
*********************************

The Drive should be used when you want to easily share data between components but the Path enables to create cleaner shareable
component where you want to exposes some files to be transferred (like an HPO component sharing the best model weights) for anyone else to use.

The Drive is more intuitive and easier to get on-boarded with, but in more advanced use cases, you might appreciate the Path Object
which makes uni-directional files transfer simpler.

----

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: The Drive Object.
   :description: Put, List and Get Files From a Shared Drive Disk.
   :col_css: col-md-4
   :button_link: drive.html
   :height: 180
   :tag: Basic

.. displayitem::
   :header: The Path Object.
   :description: Transfer Files From One Component to Another by Reference.
   :col_css: col-md-4
   :button_link: path.html
   :height: 180
   :tag: Intermediate

.. raw:: html

        </div>
    </div>
