:orphan:

***********************************
3. Build the Load Testing Component
***********************************

Now, we are going to create a component to test the performance of your model server.

We are going to use a python performance testing tool called `Locust <https://github.com/locustio/locust>`_.

.. literalinclude:: ./locust_component.py


Finally, once the component is done, we need to create a ``locustfile.py`` file which defines the format of the request to send to your model server.

The endpoint to hit has the following format: ``/v2/models/{MODEL_NAME}/versions/{VERSION}/infer``.

.. literalinclude:: ./locustfile.py


----

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: 1.  Build a Train Component
   :description: Train a model and store its checkpoints with SKlearn
   :col_css: col-md-4
   :button_link: train.html
   :height: 150
   :tag: Intermediate

.. displayitem::
   :header: 2. Build a Model Server Component
   :description: Use MLServer to server your models
   :col_css: col-md-4
   :button_link: model_server.html
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
