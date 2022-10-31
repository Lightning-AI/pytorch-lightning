###################################
Add a web UI with Streamlit (basic)
###################################
**Audience:** Users who want to add a web UI written with Python.

**Prereqs:** Basic python knowledge.

----

******************
What is Streamlit?
******************
Streamlit is a web user interface builder for Python developers. Streamlit builds beautiful web pages
directly from Python.

Install Streamlit with:

.. code:: bash

    pip install streamlit

----

*************************
Run a basic streamlit app
*************************

..
    To explain how to use Streamlit with Lightning, let's replicate the |st_link|.

    .. |st_link| raw:: html

       <a href="https://01g3p9day7x7fcjtc3h50h1hfg.litng-ai-03.litng.ai/view/home" target="_blank">example running here</a>

In the next few sections we'll build an app step-by-step.
First **create a file named app.py** with the app content:

.. code:: python

    # app.py
    import lightning as L
    import lightning.app.frontend as frontend
    import streamlit as st

    def your_streamlit_app(lightning_app_state):
        st.write('hello world')

    class LitStreamlit(L.LightningFlow):
        def configure_layout(self):
            return frontend.StreamlitFrontend(render_fn=your_streamlit_app)

    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_streamlit = LitStreamlit()

        def run(self):
            self.lit_streamlit.run()

        def configure_layout(self):
            tab1 = {"name": "home", "content": self.lit_streamlit}
            return tab1

    app = L.LightningApp(LitApp())

add "streamlit" to a requirements.txt file:

.. code:: bash

    echo 'streamlit' >> requirements.txt

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

0. Define a streamlit app
^^^^^^^^^^^^^^^^^^^^^^^^^
First, find the streamlit app you want to integrate. In this example, that app looks like:

.. code:: python

    import streamlit as st

    def your_streamlit_app():
        st.write('hello world')

Refer to the `Streamlit documentation <https://docs.streamlit.io/>`_ for more complex examples.

----

1. Add Streamlit to a component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Link this function to the Lightning App by using the ``StreamlitFrontend`` class which needs to be returned from
the ``configure_layout`` method of the Lightning component you want to connect to Streamlit.

.. code:: python
    :emphasize-lines: 9-11

    # app.py
    import lightning as L
    import lightning.app.frontend as frontend
    import streamlit as st

    def your_streamlit_app(lightning_app_state):
        st.write('hello world')

    class LitStreamlit(L.LightningFlow):
        def configure_layout(self):
            return frontend.StreamlitFrontend(render_fn=your_streamlit_app)

    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_streamlit = LitStreamlit()

        def run(self):
            self.lit_streamlit.run()

        def configure_layout(self):
            tab1 = {"name": "home", "content": self.lit_streamlit}
            return tab1

    app = L.LightningApp(LitApp())

The ``render_fn`` argument of the ``StreamlitFrontend`` class, points to a function that runs your Streamlit app.
The first argument to the function is the lightning app state. Any changes to the app state update the app.

----

2. Route the UI in the root component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The second step, is to tell the Root component in which tab to render this component's UI.
In this case, we render the ``LitStreamlit`` UI in the ``home`` tab of the application.

.. code:: python
    :emphasize-lines: 22

    # app.py
    import lightning as L
    import lightning.app.frontend as frontend
    import streamlit as st

    def your_streamlit_app(lightning_app_state):
        st.write('hello world')

    class LitStreamlit(L.LightningFlow):
        def configure_layout(self):
            return frontend.StreamlitFrontend(render_fn=your_streamlit_app)

    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_streamlit = LitStreamlit()

        def run(self):
            self.lit_streamlit.run()

        def configure_layout(self):
            tab1 = {"name": "home", "content": self.lit_streamlit}
            return tab1

    app = L.LightningApp(LitApp())
