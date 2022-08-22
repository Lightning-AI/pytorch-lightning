:orphan:

*************************************
2. Develop the Model Server Component
*************************************

In the code below, we use `MLServer <https://github.com/SeldonIO/MLServer>`_ which aims to provide an easy way to start serving your machine learning models through a REST and gRPC interface,
fully compliant with KFServing's V2 Dataplane spec.

.. literalinclude:: ./model_server.py

----

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: 1.  Develop a Train Component
   :description: Train a model and store its checkpoints with SKlearn
   :col_css: col-md-4
   :button_link: train.html
   :height: 150
   :tag: Intermediate

.. displayitem::
   :header: 3. Develop a Load Testing Component
   :description: Use Locust to test your model servers
   :col_css: col-md-4
   :button_link: load_testing.html
   :height: 150
   :tag: Intermediate

.. displayitem::
   :header: 4. Putting everything together.
   :description: Ensemble the Components together and run the App
   :col_css: col-md-4
   :button_link: putting_everything_together.html
   :height: 150
   :tag: basic

.. raw:: html

        </div>
    </div>
