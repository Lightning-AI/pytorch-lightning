############################################
Develop a Lightning Component (intermediate)
############################################

**Audience:** Users who want to connect a UI to a Lightning Component (Component).

----

*****************************
Add a web user interface (UI)
*****************************
Every lightning component can have its own user interface (UI). Lightning components support any kind
of UI interface such as dash, gradio, panel, react.js, streamlit, vue.js, web urls,
etc...(:doc:`full list here <../add_web_ui/index>`).

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

To *connect* this user interface to the Component, define the configure_layout method:

.. code:: python
    :emphasize-lines: 5, 6

    import lightning as L


    class LitHTMLComponent(L.LightningFlow):
        def configure_layout(self):
            return L.app.frontend.StaticWebFrontend(serve_dir="path/to/folder/with/index.html/inside")

Finally, route the Component's UI through the root Component's **configure_layout** method:

.. code:: python
    :emphasize-lines: 14

    # app.py
    import lightning as L


    class LitHTMLComponent(L.LightningFlow):
        def configure_layout(self):
            return L.app.frontend.StaticWebFrontend(serve_dir="path/to/folder/with/index.html/inside")


    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_html_component = LitHTMLComponent()

        def configure_layout(self):
            tab1 = {"name": "home", "content": self.lit_html_component}
            return tab1


    app = L.LightningApp(LitApp())

Run your App and you'll see the UI on the Lightning App view:

.. code:: bash

    lightning run app app.py
