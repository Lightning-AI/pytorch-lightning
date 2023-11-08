:orphan:

######################################
Add a web UI with Panel (intermediate)
######################################

**Audience:** Users who want to communicate between the Lightning App and Panel.

**Prereqs:** Must have read the :doc:`Panel basic <basic>` guide.

----

**************************************
Interact with the Component from Panel
**************************************

The ``PanelFrontend`` enables user interactions with the Lightning App using widgets.
You can modify the state variables of a Lightning Component using the ``AppStateWatcher``.

For example, here we increase the ``count`` variable of the Lightning Component every time a user
presses a button:

.. code:: python

    # app_panel.py

    import panel as pn
    from lightning.app.frontend import AppStateWatcher

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



.. code:: python

    # app.py

    import lightning as L
    from lightning.app.frontend import PanelFrontend

    class LitPanel(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.count = 0
            self.last_count = 0

        def run(self):
            if self.count != self.last_count:
                self.last_count = self.count
                print("Count changed to: ", self.count)

        def configure_layout(self):
            return PanelFrontend("app_panel.py")


    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_panel = LitPanel()

        def run(self):
            self.lit_panel.run()

        def configure_layout(self):
            return {"name": "home", "content": self.lit_panel}


    app = L.LightningApp(LitApp())

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/panel-lightning-counter-from-frontend.gif
   :alt: Panel Lightning App updating a counter from the frontend

   Panel Lightning App updating a counter from the frontend

----

************************************
Interact with Panel from a Component
************************************

To update the `PanelFrontend` from any Lightning Component, update the property in the Component.
Make sure to call the ``run`` method from the parent component.

In this example, we update the ``count`` value of the Component:

.. code:: python

    # app_panel.py

    import panel as pn
    from lightning.app.frontend import AppStateWatcher

    app = AppStateWatcher()

    pn.extension(sizing_mode="stretch_width")

    def counter(state):
        return f"Counter: {state.count}"

    last_update = pn.bind(counter, app.param.state)

    pn.panel(last_update).servable()

.. code:: python

    # app.py

    from datetime import datetime as dt
    from lightning.app.frontend import PanelFrontend

    import lightning as L


    class LitPanel(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.count = 0
            self._last_update = dt.now()

        def run(self):
            now = dt.now()
            if (now - self._last_update).microseconds >= 250:
                self.count += 1
                self._last_update = now
                print("Counter changed to: ", self.count)

        def configure_layout(self):
            return PanelFrontend("app_panel.py")


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

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/panel-lightning-counter-from-component.gif
   :alt: Panel Lightning App updating a counter from the component

   Panel Lightning App updating a counter from the Component

----

*************
Tips & Tricks
*************

* Caching: Panel provides the easy to use ``pn.state.cache`` memory based, ``dict`` caching. If you are looking for something persistent try `DiskCache <https://grantjenks.com/docs/diskcache/>`_ its really powerful and simple to use. You can use it to communicate large amounts of data between the components and frontend(s).

* Notifications: Panel provides easy to use `notifications <https://blog.holoviz.org/panel_0.13.0.html#Notifications>`_. You can for example use them to provide notifications about runs starting or ending.

* Tabulator Table: Panel provides the `Tabulator table <https://blog.holoviz.org/panel_0.13.0.html#Expandable-rows>`_ which features expandable rows. The table is useful to provide for example an overview of you runs. But you can dig into the details by clicking and expanding the row.

* Task Scheduling: Panel provides easy to use `task scheduling <https://blog.holoviz.org/panel_0.13.0.html#Task-scheduling>`_. You can use this to for example read and display files created by your components on a scheduled basis.

* Terminal: Panel provides the `Xterm.js terminal <https://panel.holoviz.org/reference/widgets/Terminal.html>`_ which can be used to display live logs from your components and allow you to provide a terminal interface to your component.

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/panel-lightning-github-runner.gif
   :alt: Panel Lightning App running models on github

   Panel Lightning App running models on GitHub

----

**********
Next Steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Add a web user interface (UI)
   :description: Users who want to add a UI to their Lightning Apps
   :col_css: col-md-6
   :button_link: ../index.html
   :height: 150
   :tag: intermediate

.. raw:: html

        </div>
    </div>
