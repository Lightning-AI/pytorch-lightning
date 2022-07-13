######################################
Add a web UI with Panel (intermediate)
######################################

**Audience:** Users who want to communicate between the Lightning App and Panel.

**Prereqs:** Must have read the `panel basic <basic.html>`_ guide.

----

**************************************
Interact with the component from Panel
**************************************

The ``PanelFrontend`` enables user interactions with the Lightning App via widgets.
You can modify the state variables of a Lightning component via the ``AppStateWatcher``.

For example, here we increase the ``count`` variable of the Lightning Component every time a user
presses a button:

.. code:: bash
    
    # app_panel.py

    import panel as pn
    from lightning_app.frontend.panel import AppStateWatcher

    pn.extension(sizing_mode="stretch_width")

    app = AppStateWatcher()

    submit_button = pn.widgets.Button(name="submit")

    @pn.depends(submit_button, watch=True)
    def submit(_):
        app.state.count += 1

    @pn.depends(app.param.state)
    def current_count(_):
        return f"current count: {app.state.count}"

    pn.Column(
        submit_button,
        current_count,
    ).servable()



.. code:: bash
    
    # app.py

    import lightning as L
    from lightning_app.frontend.panel import PanelFrontend

    class LitPanel(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self._frontend = PanelFrontend("app_panel.py")
            self.count = 0
            self._last_count=0

        def configure_layout(self):
            return self._frontend

        def run(self):
            if self.count != self._last_count:
                self._last_count=self.count
                print("Count changed to: ", self.count)


    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_panel = LitPanel()

        def run(self):
            self.lit_panel.run()

        def configure_layout(self):
            return {"name": "home", "content": self.lit_panel}


    app = L.LightningApp(LitApp())

.. figure:: https://raw.githubusercontent.com/MarcSkovMadsen/awesome-panel-assets/master/videos/panel-lightning/panel-lightning-counter-from-frontend.gif
   :alt: Panel Lightning App updating a counter from the frontend

   Panel Lightning App updating a counter from the frontend

----

**************************************
Interact with Panel from the component
**************************************

To update the `PanelFrontend` from any Lightning component, update the property in the component. Make sure to call ``run`` method from the
parent component.

In this example we update the value of ``count`` from the component:

.. code:: bash

    # app_panel.py

    import panel as pn
    from lightning_app.frontend.panel import AppStateWatcher

    app=AppStateWatcher()

    pn.extension(sizing_mode="stretch_width")

    def counter(state):
        return f"Counter: {state.count}"

    last_update = pn.bind(counter, app.param.state)

    pn.panel(last_update).servable()

.. code:: bash

    # app.py

    from datetime import datetime as dt
    from lightning_app.frontend.panel import PanelFrontend

    import lightning_app as L


    class LitPanel(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self._frontend = PanelFrontend("app_panel.py")
            self.count = 0
            self._last_update=dt.now()

        def configure_layout(self):
            return self._frontend

        def run(self):
            now = dt.now()
            if (now-self._last_update).microseconds>=250:
                self.count += 1
                self._last_update = now
                print("Counter changed to: ", self.count)

    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_panel = LitPanel()

        def run(self):
            self.lit_panel.run()

        def configure_layout(self):
            tab1 = {"name": "home", "content": self.lit_panel}
            return tab1

    app = L.LightningApp(LitApp())

.. figure:: https://raw.githubusercontent.com/MarcSkovMadsen/awesome-panel-assets/master/videos/panel-lightning/panel-lightning-counter-from-component.gif
   :alt: Panel Lightning App updating a counter from the component

   Panel Lightning App updating a counter from the component