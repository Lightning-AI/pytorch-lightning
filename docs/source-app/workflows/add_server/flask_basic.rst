###############################
Add a web UI with Flask (basic)
###############################
**Audience:** Users who want to enable a flask app within a component.

**Prereqs:** Basic python knowledge.

----

**************
What is Flask?
**************
Flask is a web framework, that lets you develop web applications in Python easily.

----

************************
Add Flask to a component
************************
First, define your flask app as you normally would without Lightning:

.. code:: python
    :emphasize-lines: 9

    from flask import Flask

    flask_app = Flask(__name__)


    @flask_app.route("/")
    def hello():
        return "Hello, World!"


    flask_app.run(host="0.0.0.0", port=80)

To enable the server inside the component, start the Flask server in the run method and use the ``self.host`` and ``self.port`` properties:

.. code:: python
    :emphasize-lines: 12

    import lightning as L
    from flask import Flask


    class LitFlask(L.LightningWork):
        def run(self):
            flask_app = Flask(__name__)

            @flask_app.route("/")
            def hello():
                return "Hello, World!"

            flask_app.run(host=self.host, port=self.port)

----

**************************************
Route the server in the root component
**************************************
The final step, is to tell the Root component in which tab to render this component's output:
In this case, we render the ``LitFlask`` output in the ``home`` tab of the application.

.. code:: python
    :emphasize-lines: 17, 23

    import lightning as L
    from flask import Flask


    class LitFlask(L.LightningWork):
        def run(self):
            flask_app = Flask(__name__)

            @flask_app.route("/")
            def hello():
                return "Hello, World!"

            flask_app.run(host=self.host, port=self.port)


    class Root(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_flask = LitFlask(parallel=True)

        def run(self):
            self.lit_flask.run()

        def configure_layout(self):
            tab1 = {"name": "home", "content": self.lit_flask}
            return tab1


    app = L.LightningApp(Root())

We use the ``parallel=True`` argument of ``LightningWork`` to run the server in the background
while the rest of the Lightning App runs everything else.

----

***********
Run the app
***********
Start the app to see your new UI!

.. code:: bash

    lightning run app app.py

To run the app on the cloud, use the ``--cloud`` argument.

.. code:: bash

    lightning run app app.py --cloud

----

********
Examples
********
Here are a few example apps that expose a Flask server via a component:

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Example 1
   :description: TODO
   :col_css: col-md-4
   :button_link: example_app.html
   :height: 150

.. displayitem::
   :header: Example 2
   :description: TODO
   :col_css: col-md-4
   :button_link: example_app.html
   :height: 150

.. displayitem::
   :header: Example 3
   :description: TODO
   :col_css: col-md-4
   :button_link: example_app.html
   :height: 150

.. raw:: html

        </div>
    </div>
