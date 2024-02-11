#########################
Enable any server (basic)
#########################
**Audience:** Users who want to enable an arbitrary server/UI.

**Prereqs:** Basic python knowledge.

----

*****************
What is a server?
*****************
A server is a program that enables other programs or users to connect to it. As long as your server can listen on a port,
you can enable it with a Lightning App.

----

***************************
Add a server to a component
***************************
Any server that listens on a port, can be enabled via a work. For example, here's a plain python server:

.. code:: python
    :emphasize-lines: 11-12

    import socketserver
    from http import HTTPStatus, server


    class PlainServer(server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            # Data must be passed as bytes to the `self.wfile.write` call
            html = b"<h1 style='color: blue'> Hello lit world </div>"
            self.wfile.write(html)


    httpd = socketserver.TCPServer(("localhost", "3000"), PlainServer)
    httpd.serve_forever()

To enable the server inside the component, start the server in the run method and use the ``self.host`` and ``self.port`` properties:

.. code:: python
    :emphasize-lines: 14-15

    import lightning as L
    import socketserver
    from http import HTTPStatus, server


    class PlainServer(server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            # Data must be passed as bytes to the `self.wfile.write` call
            html = b"<h1 style='color: blue'> Hello lit world </div>"
            self.wfile.write(html)


    class LitServer(L.LightningWork):
        def run(self):
            httpd = socketserver.TCPServer((self.host, self.port), PlainServer)
            httpd.serve_forever()

----

**************************************
Route the server in the root component
**************************************
The final step, is to tell the Root component in which tab to render this component's output:
In this case, we render the ``LitServer`` output in the ``home`` tab of the application.

.. code:: python
    :emphasize-lines: 20, 23, 28

    import lightning as L
    import socketserver
    from http import HTTPStatus, server


    class PlainServer(server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            # Data must be passed as bytes to the `self.wfile.write` call
            html = b"<h1 style='color: blue'> Hello lit world </div>"
            self.wfile.write(html)


    class LitServer(L.LightningWork):
        def run(self):
            httpd = socketserver.TCPServer((self.host, self.port), PlainServer)
            httpd.serve_forever()


    class Root(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_server = LitServer(parallel=True)

        def run(self):
            self.lit_server.run()

        def configure_layout(self):
            tab1 = {"name": "home", "content": self.lit_server}
            return tab1


    app = L.LightningApp(Root())

We use the ``parallel=True`` argument of ``LightningWork`` to run the server in parallel
while the rest of the Lightning App runs everything else.

----

***********
Run the app
***********
Start the app to see your new UI!

.. code:: bash

    lightning_app run app app.py

To run the app on the cloud, use the ``--cloud`` argument.

.. code:: bash

    lightning_app run app app.py --cloud

----

*****************************************
Interact with a component from the server
*****************************************

.. TODO:: how do we do this?


----

*****************************************
Interact with the server from a component
*****************************************

.. TODO:: how do we do this?

----

********
Examples
********
Here are a few example apps that expose a server via a component:

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Example: Tensorboard
   :description: TODO
   :col_css: col-md-4
   :button_link: example_app.html
   :height: 150

.. displayitem::
   :header: Example: Streamlit
   :description: TODO
   :col_css: col-md-4
   :button_link: example_app.html
   :height: 150

.. displayitem::
   :header: Example: React
   :description: TODO
   :col_css: col-md-4
   :button_link: example_app.html
   :height: 150

.. raw:: html

        </div>
    </div>
