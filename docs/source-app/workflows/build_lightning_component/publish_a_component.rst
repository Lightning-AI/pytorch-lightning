#############################
Publish a Lightning component
#############################
**Audience:** Users who want to build a component to publish to the Lightning Gallery

----

********************************
Generate component from template
********************************
The fastest way to build a component that is ready to be published to the component Gallery is to use
the default template.

Generate your component template with this command:

.. code:: bash

    lightning init component your-component-name

----

*****************
Run the component
*****************
To test that your component works, first install all dependencies:

.. code:: bash

    cd your-component
    pip install -r requirements.txt
    pip install -e .

Now import your component and use it in an app:

.. code:: python

    # app.py
    from your_component import TemplateComponent
    import lightning_app as la


    class LitApp(lapp.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.your_component = TemplateComponent()

        def run(self):
            print("this is a simple Lightning app to verify your component is working as expected")
            self.your_component.run()


    app = lapp.LightningApp(LitApp())

and run the app:

.. code:: bash

    lightning run app app.py
