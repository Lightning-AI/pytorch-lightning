################################
Add a web UI with Gradio (basic)
################################
**Audience:** Users who want to add a web UI written with Python.

**Prereqs:** Basic python knowledge.

----

***************
What is Gradio?
***************
Gradio is a Python library that automatically generates a web interface to demo a machine learning model.

----

*****************
Install gradio
*****************
First, install gradio.

.. code:: bash

    pip install gradio

----

**************************
Create the gradio demo app
**************************
To explain how to use Gradio with Lightning, let's replicate the |gradio_link|.

.. |gradio_link| raw:: html

   <a href="https://01g3p7np2v3far1jg6ccbvppah.litng-ai-03.litng.ai/view/home" target="_blank">example running here</a>

In the next few sections we'll build an app step-by-step.
First **create a file named app.py** with the app content:

.. code:: python

    import lightning as L
    from lightning.app.components import ServeGradio
    import gradio as gr

    class LitGradio(ServeGradio):

        inputs = gr.inputs.Textbox(default='lightning', label='name input')
        outputs = gr.outputs.Textbox(label='output')
        examples = [["hello lightning"]]

        def predict(self, input_text):
            return self.model(input_text)

        def build_model(self):
            fake_model = lambda x: f"hello {x}"
            return fake_model

    class RootFlow(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_gradio = LitGradio()

        def run(self):
            self.lit_gradio.run()

        def configure_layout(self):
            return [{"name": "home", "content": self.lit_gradio}]

    app = L.LightningApp(RootFlow())

add "gradio" to a requirements.txt file:

.. code:: bash

    echo 'gradio' >> requirements.txt

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

Create a Gradio component
^^^^^^^^^^^^^^^^^^^^^^^^^
To create a Gradio component, simply take any Gradio app and subclass it from ``ServeGradio``.
If you haven't created a Gradio demo, you have to implement the following elements:

1. Input which is text.
2. Output which is text.
3. A build_model function.
4. A predict function.

|

Here's an example:

.. code:: python
    :emphasize-lines: 4

    from lightning.app.components import ServeGradio
    import gradio as gr

    class LitGradio(ServeGradio):

        inputs = gr.inputs.Textbox(default='lightning', label='name input')
        outputs = gr.outputs.Textbox(label='output')

        def predict(self, input_text):
            return self.model(input_text)

        def build_model(self):
            fake_model = lambda x: f"hello {x}"
            return fake_model

This fake model simply concatenates 2 strings.

----

Route the UI in the root component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now, tell the Root component in which tab to render this component's UI.
In this case, we render the ``LitGradio`` UI in the ``home`` tab of the application.

.. code:: python
    :emphasize-lines: 21, 27

    import lightning as L
    from lightning.app.components import ServeGradio
    import gradio as gr

    class LitGradio(ServeGradio):

        inputs = gr.inputs.Textbox(default='lightning', label='name input')
        outputs = gr.outputs.Textbox(label='output')
        examples = [["hello lightning"]]

        def predict(self, input_text):
            return self.model(input_text)

        def build_model(self):
            fake_model = lambda x: f"hello {x}"
            return fake_model

    class RootFlow(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_gradio = LitGradio()

        def run(self):
            self.lit_gradio.run()

        def configure_layout(self):
            return [{"name": "home", "content": self.lit_gradio}]

    app = L.LightningApp(RootFlow())

----

Call run
^^^^^^^^
Finally, don't forget to call run inside the Root Flow to serve the Gradio app.

.. code:: python
    :emphasize-lines: 24

    import lightning as L
    from lightning.app.components import ServeGradio
    import gradio as gr

    class LitGradio(ServeGradio):

        inputs = gr.inputs.Textbox(default='lightning', label='name input')
        outputs = gr.outputs.Textbox(label='output')
        examples = [["hello lightning"]]

        def predict(self, input_text):
            return self.model(input_text)

        def build_model(self):
            fake_model = lambda x: f"hello {x}"
            return fake_model

    class RootFlow(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_gradio = LitGradio()

        def run(self):
            self.lit_gradio.run()

        def configure_layout(self):
            return [{"name": "home", "content": self.lit_gradio}]

    app = L.LightningApp(RootFlow())
