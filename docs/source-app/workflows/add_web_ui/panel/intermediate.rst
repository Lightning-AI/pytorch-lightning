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

*******************
Panel Tips & Tricks
*******************

- Caching: Panel provides the easy to use ```pn.state.cache` memory based, ``dict`` caching. If you are looking for something persistent try `DiskCache <https://grantjenks.com/docs/diskcache/>`_ its really powerful and simple to use. You can use it to communicate large amounts of data between the components and frontend(s).
- Notifactions: Panel provides easy to use `notifications <https://blog.holoviz.org/panel_0.13.0.html#Notifications>`_. You can for example use them to provide notifications about runs starting or ending.
- Tabulator Table: Panel provides the `Tabulator table <https://blog.holoviz.org/panel_0.13.0.html#Expandable-rows>`_ which features expandable rows. The table is useful to provide for example an overview of you runs. But you can dig into the details by clicking and expanding the row.
- Task Scheduling: Panel provides easy to use `task scheduling <https://blog.holoviz.org/panel_0.13.0.html#Task-scheduling>`. You can use this to for example read and display files created by your components on a schedule basis.
- Terminal: Panel provides the `Xterm.js terminal <https://panel.holoviz.org/reference/widgets/Terminal.html>`_ which can be used to display live logs from your components and allow you to provide a terminal interface to your component.

.. figure:: https://raw.githubusercontent.com/MarcSkovMadsen/awesome-panel-assets/master/videos/panel-lightning/panel-lightning-github-runner.gif
   :alt: Panel Lightning App running models on github

   Panel Lightning App running models on github

# Todo: Add link to the code and running app.