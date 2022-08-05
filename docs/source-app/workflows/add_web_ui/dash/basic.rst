##############################
Add a web UI with Dash (basic)
##############################
**Audience:** Users who want to add a web UI with Dash by Plotly.

**Prereqs:** Basic python knowledge.

----

*************
What is Dash?
*************
`Dash <https://plotly.com/dash/>`_ is the original low-code framework for rapidly building data apps in Python, R, Julia, and F# (experimental).

Install Dash with:

.. code:: bash

    pip install dash

----

************************
Create the dash demo app
************************

To explain how to use Dash with Lightning, let's build a simple app with Dash.


..
    To explain how to use Dash with Lightning, let's replicate the |dash_link|.

    .. |dash_link| raw:: html

       <a href="https://01g3p4bf3m61xsm2yzn0966q59.litng-ai-03.litng.ai/view/home" target="_blank">example running here</a>

In the next few sections we'll build an app step-by-step.
First **create a file named app.py** with the app content:

.. code:: bash

        import lightning as L
        import dash
        import plotly.express as px

        class LitDash(L.LightningWork):
            def run(self):
                dash_app = dash.Dash(__name__)
                X = [1, 2, 3, 4, 5, 6]
                Y = [2, 4, 8, 16, 32, 64]
                fig = px.line(x=X, y=Y)

                dash_app.layout = dash.html.Div(children=[
                    dash.html.H1(children='⚡ Hello Dash + Lightning⚡'),
                    dash.html.Div(children='The Dash framework running inside a ⚡ Lightning App'),
                    dash.dcc.Graph(id='example-graph', figure=fig)
                ])

                dash_app.run_server(host=self.host, port=self.port)

        class LitApp(L.LightningFlow):
            def __init__(self):
                super().__init__()
                self.lit_dash = LitDash(parallel=True)

            def run(self):
                self.lit_dash.run()

            def configure_layout(self):
                tab1 = {"name": "home", "content": self.lit_dash}
                return tab1

        app = L.LightningApp(LitApp())


add 'dash' to a requirements.txt file:

.. code:: bash

    echo "dash" >> requirements.txt

this is a best practice to make apps reproducible.

----

***********
Run the app
***********
Run the app locally to see it!

.. code:: python

    lightning run app app.py

Now run it on the cloud as well:

.. code:: python

    lightning run app app.py --cloud

----

************************
Step-by-step walkthrough
************************
In this section, we explain each part of this code in detail.

----

0. Define a Dash app
^^^^^^^^^^^^^^^^^^^^
First, find the dash app you want to integrate. In this example, that app looks like:

.. code:: python

        import dash
        import plotly.express as px

        dash_app = dash.Dash(__name__)
        X = [1, 2, 3, 4, 5, 6]
        Y = [2, 4, 8, 16, 32, 64]
        fig = px.line(x=X, y=Y)

        dash_app.layout = dash.html.Div(children=[
            dash.html.H1(children='⚡ Hello Dash + Lightning⚡'),
            dash.html.Div(children='The Dash framework running inside a ⚡ Lightning App'),
            dash.dcc.Graph(id='example-graph', figure=fig)
        ])

        dash_app.run_server(host='0.0.0.0', port=80)

This dash app plots a simple line curve along with some HTMlapp.
`Visit the Dash documentation for the full API <https://plotly.com/dash/>`_.

----

1. Add Dash to a component
^^^^^^^^^^^^^^^^^^^^^^^^^^
Add the dash app to the run method of a ``LightningWork`` component and run the server on that component's **host** and **port**:

.. code:: python
    :emphasize-lines: 6, 18

        import lightning as L
        import dash
        import plotly.express as px

        class LitDash(L.LightningWork):
            def run(self):
                dash_app = dash.Dash(__name__)
                X = [1, 2, 3, 4, 5, 6]
                Y = [2, 4, 8, 16, 32, 64]
                fig = px.line(x=X, y=Y)

                dash_app.layout = dash.html.Div(children=[
                    dash.html.H1(children='⚡ Hello Dash + Lightning⚡'),
                    dash.html.Div(children='The Dash framework running inside a ⚡ Lightning App'),
                    dash.dcc.Graph(id='example-graph', figure=fig)
                ])

                dash_app.run_server(host=self.host, port=self.port)

        class LitApp(L.LightningFlow):
            def __init__(self):
                super().__init__()
                self.lit_dash = LitDash(parallel=True)

            def run(self):
                self.lit_dash.run()

            def configure_layout(self):
                tab1 = {"name": "home", "content": self.lit_dash}
                return tab1

        app = L.LightningApp(LitApp())

----

2. Route the UI in the root component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The final step, is to tell the Root component in which tab to render this component's UI.
In this case, we render the ``LitDash`` UI in the ``home`` tab of the application.

.. code:: python
    :emphasize-lines: 23, 29

        import lightning as L
        import dash
        import plotly.express as px

        class LitDash(L.LightningWork):
            def run(self):
                dash_app = dash.Dash(__name__)
                X = [1, 2, 3, 4, 5, 6]
                Y = [2, 4, 8, 16, 32, 64]
                fig = px.line(x=X, y=Y)

                dash_app.layout = dash.html.Div(children=[
                    dash.html.H1(children='⚡ Hello Dash + Lightning⚡'),
                    dash.html.Div(children='The Dash framework running inside a ⚡ Lightning App'),
                    dash.dcc.Graph(id='example-graph', figure=fig)
                ])

                dash_app.run_server(host=self.host, port=self.port)

        class LitApp(L.LightningFlow):
            def __init__(self):
                super().__init__()
                self.lit_dash = LitDash(parallel=True)

            def run(self):
                self.lit_dash.run()

            def configure_layout(self):
                tab1 = {"name": "home", "content": self.lit_dash}
                return tab1

        app = L.LightningApp(LitApp())

We use the ``parallel=True`` argument of ``LightningWork`` to run the server in the background
while the rest of the Lightning App runs everything else.
