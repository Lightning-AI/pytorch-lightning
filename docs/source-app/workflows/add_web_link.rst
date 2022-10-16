##############
Add a web link
##############
**Audience:** Users who want to link to other pages from their app.

----

**************
Add a url link
**************
In this example we'll replicate |urls_link|.

To add a url link to an app, simply specify it in the ``configure_layout`` method
and connect the UIs. Create a file named **app.py** with this code:

.. |urls_link| raw:: html

   <a href="https://01g3ppzb6t3pp5cf1zhg8shpym.litng-ai-03.litng.ai/view/TB%20logs" target="_blank">the app running here</a>

.. code:: python
    :emphasize-lines: 7,11

    import lightning as L

    class LitApp(L.LightningFlow):
        def configure_layout(self):
            tab_1 = {
                "name": "Logger",
                "content": "https://bit.ly/tb-aasae"
            }
            tab_2 = {
                "name": "Paper",
                "content": "https://arxiv.org/pdf/2107.12329.pdf"
            }
            return tab_1, tab_2

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
