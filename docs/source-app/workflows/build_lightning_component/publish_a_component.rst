#############################
Publish a Lightning Component
#############################

**Audience:** Users who want to build a Ligthtning Component (Component) to publish to the Lightning Gallery

----

***********************************
Develop a Component from a template
***********************************

The fastest way to build a Component that is ready to be published to the component Gallery is to use
the default template.

Generate your Component template with this command:

.. code:: python

    lightning init component your-component-name

----

*****************
Run the Component
*****************

To test that your Component works, first install all dependencies:

.. code:: bash

    cd your-component
    pip install -r requirements.txt
    pip install -e .

Now import your Component and use it in a Lightning App:

.. code:: python

    # app.py
    from your_component import TemplateComponent
    import lightning as L

    class LitApp(L.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.your_component = TemplateComponent()

        def run(self):
            print('this is a simple Lightning app to verify your component is working as expected')
            self.your_component.run()

    app = L.LightningApp(LitApp())

and run the app:

.. code:: bash

    lightning run app app.py
