########################
Arrange app tabs (basic)
########################
**Audience:** Users who want to control the layout of their app user interface.

----

*****************************
Enable a full-page single tab
*****************************

To enable a single tab on the app UI, return a single dictionary from the ``configure_layout`` method:

.. code:: python
    :emphasize-lines: 9

    import lightning as L


    class DemoComponent(L.demo.dumb_component):
        def configure_layout(self):
            tab1 = {"name": "THE TAB NAME", "content": self.component_a}
            return tab1


    app = L.LightningApp(DemoComponent())


The "name" key defines the visible name of the tab on the UI. It also shows up in the URL.
The **"content"** key defines the target component to render in that tab.
When returning a single tab element like shown above, the UI will display it in full-page mode.


----

********************
Enable multiple tabs
********************

.. code:: python
    :emphasize-lines: 7

    import lightning as L


    class DemoComponent(L.demo.dumb_component):
        def configure_layout(self):
            tab1 = {"name": "Tab A", "content": self.component_a}
            tab2 = {"name": "Tab B", "content": self.component_b}
            return tab1, tab2


    app = L.LightningApp(DemoComponent())

The order matters! Try any of the following configurations:

.. code:: python
    :emphasize-lines: 4, 9

    def configure_layout(self):
        tab1 = {"name": "Tab A", "content": self.component_a}
        tab2 = {"name": "Tab B", "content": self.component_b}
        return tab1, tab2


    def configure_layout(self):
        tab1 = {"name": "Tab A", "content": self.component_a}
        tab2 = {"name": "Tab B", "content": self.component_b}
        return tab2, tab1
