##########################################
Build a Lightning component (intermediate)
##########################################
**Audience:** Users who want to connect a UI to a Lightning component.

----

*****************************
Add a web user interface (UI)
*****************************
Every lightning component can have its own user interface (UI). Lightning components support any kind
of UI interface such as react.js, vue.js, streamlit, gradio, dash, web urls, etc...(`full list here <../add_web_ui/index.html>`_).

Let's say that we have a user interface defined in html:

.. code:: html

    <!--index.html-->
    <html>
    <head>
    </head>
    <body>
        <h1>Hello World<h1>
    </body>
    </html>

To *connect* this user interface to the component, define the configure_layout method:

.. code:: python
    :emphasize-lines: 5, 6

    import lightning_app as la
    from lightning_app.frontend.web import StaticWebFrontend


    class LitHTMLComponent(lapp.LightningFlow):
        def configure_layout(self):
            return StaticWebFrontend(serve_dir="path/to/folder/with/index.html/inside")

Finally, route the component's UI through the root component's **configure_layout** method:

.. code:: python
    :emphasize-lines: 14

    # app.py
    import lightning_app as la


    class LitHTMLComponent(lapp.LightningFlow):
        def configure_layout(self):
            return lapp.frontend.web.StaticWebFrontend(serve_dir="path/to/folder/with/index.html/inside")


    class LitApp(lapp.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_html_component = LitHTMLComponent()

        def configure_layout(self):
            tab1 = {"name": "home", "content": self.lit_html_component}
            return tab1


    app = lapp.LightningApp(LitApp())

Run your app and you'll see the UI on the Lightning App view:

.. code:: bash

    lightning run app app.py
