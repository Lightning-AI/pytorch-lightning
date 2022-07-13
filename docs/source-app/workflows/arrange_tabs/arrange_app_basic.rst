########################
Arrange app tabs (basic)
########################
**Audience:** Users who want to control the layout of their app user interface.

----

*******************
Enable a single tab
*******************
To enable a single tab on the app UI, return a single dictionary from the ``configure_layout`` method:

.. code:: python
    :emphasize-lines: 9

    import lightning_app as la


    class DemoComponent(lapp.demo.dumb_component):
        def configure_layout(self):
            tab1 = {"name": "THE TAB NAME", "content": self.component_a}
            return tab1


    app = lapp.LightningApp(DemoComponent())

The "name" key defines the visible name of the tab on the UI.
The **"content"** key defines the target component to render in that tab.


----

*****************************
Enable a full-page single tab
*****************************

.. code:: python
    :emphasize-lines: 6

    import lightning_app as la


    class DemoComponent(lapp.demo.dumb_component):
        def configure_layout(self):
            tab1 = {"name": None, "content": self.component_a}
            return tab1


    app = lapp.LightningApp(DemoComponent())

----

********************
Enable multiple tabs
********************

.. code:: python
    :emphasize-lines: 7

    import lightning_app as la


    class DemoComponent(lapp.demo.dumb_component):
        def configure_layout(self):
            tab1 = {"name": "Tab A", "content": self.component_a}
            tab2 = {"name": "Tab B", "content": self.component_b}
            return tab1, tab2


    app = lapp.LightningApp(DemoComponent())

order matters! Try any of the following configurations:

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
