:orphan:

****************************
1. Build the Train Component
****************************

In the code below, we create a work which trains a simple `SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_ model on the digits dataset (classification).

Once the model is trained, it is saved and a reference :class:`~lightning.app.storage.path.Path` with ``best_model_path`` state attribute.

.. literalinclude:: ./train.py

----

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: 2. Build a Model Server Component
   :description: Use MLServer to server your models
   :col_css: col-md-4
   :button_link: model_server.html
   :height: 150
   :tag: Intermediate

.. displayitem::
   :header: 3. Build a Load Testing Component
   :description: Use Locust to test your model servers
   :col_css: col-md-4
   :button_link: load_testing.html
   :height: 150
   :tag: Intermediate

.. displayitem::
   :header: 4. Putting everything together.
   :description: Ensemble the components together and run the app
   :col_css: col-md-4
   :button_link: putting_everything_together.html
   :height: 150
   :tag: basic

.. raw:: html

        </div>
    </div>
