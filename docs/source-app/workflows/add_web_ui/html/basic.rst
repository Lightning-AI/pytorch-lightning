##############################
Add a web UI with HTML (basic)
##############################
**Audience:** Users who want to add a web UI written in HTMlapp.

**Prereqs:** Basic html knowledge.

----

*************
What is HTML?
*************
HyperText Markup Language (HTML) is the Language used to create web pages. Use HTML for simple
web user interfaces that tend to be more static.

For reactive web applications, we recommend using: React.js, Angular.js or Vue.js

----

*******************
Create an HTML page
*******************
The first step is to create an HTML file named **index.html**:

.. code:: html

    <!--index.html-->
    <html>
    <head>
    </head>
    <body>
        <h1>Hello World<h1>
    </body>
    </html>

----

************************
Create the HTML demo app
************************

..
    To explain how to use html with Lightning, let's replicate the |html_app_link|.

    .. |html_app_link| raw:: html

       <a href="https://01g3pdayfptbhqfre565j8gwjr.litng-ai-03.litng.ai/view/home" target="_blank">example running here</a>

In the next few sections we'll build an app step-by-step.
First **create a file named app.py** with the app content (in the same folder as index.html):

.. code:: bash

    # app.py
    import lightning as L
    import lightning.app.frontend as frontend


    class HelloComponent(L.LightningFlow):
        def configure_layout(self):
            return frontend.StaticWebFrontend(serve_dir='.')


    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.hello_component = HelloComponent()

        def run(self):
            self.hello_component.run()

        def configure_layout(self):
            tab1 = {"name": "home", "content": self.hello_component}
            return tab1


    app = L.LightningApp(LitApp())

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

Enable an HTML UI for the component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Give the component an HTML UI, by returning a ``StaticWebFrontend`` object from the ``configure_layout`` method:

.. code:: bash
    :emphasize-lines: 6,7

    # app.py
    import lightning as L
    import lightning.app.frontend as frontend

    class HelloComponent(L.LightningFlow):
        def configure_layout(self):
            return frontend.StaticWebFrontend(serve_dir='.')

    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.hello_component = HelloComponent()

        def run(self):
            self.hello_component.run()

        def configure_layout(self):
            tab1 = {"name": "home", "content": self.hello_component}
            return tab1

    app = L.LightningApp(LitApp())

The folder path given in ``StaticWebFrontend(serve_dir=)`` must point to a folder with an ``index.html`` page.

----

Route the UI in the root component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The final step, is to tell the Root component in which tab to render this component's UI.
In this case, we render the ``HelloComponent`` UI in the ``home`` tab of the application.

.. code:: python
    :emphasize-lines: 18, 19

    # app.py
    import lightning as L
    import lightning.app.frontend as frontend

    class HelloComponent(L.LightningFlow):
        def configure_layout(self):
            return frontend.StaticWebFrontend(serve_dir='.')

    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.hello_component = HelloComponent()

        def run(self):
            self.hello_component.run()

        def configure_layout(self):
            tab1 = {"name": "home", "content": self.hello_component}
            return tab1

    app = L.LightningApp(LitApp())
