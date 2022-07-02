###############################
Add a web UI with Panel (basic)
###############################

**Audience:** Users who want to add a web UI with Panel by HoloViz.

**Prereqs:** Basic Python knowledge.

----

**************
What is Panel?
**************

`Panel`_ and the `HoloViz`_ ecosystem provides unique and powerful
features such as big data viz via `DataShader`_, easy cross filtering
via `HoloViews`_, streaming and much more.

- Panel works with the tools you know and love ❤️. Panel ties into the PyData and Jupyter ecosystems as you can develop in notebooks and use ipywidgets. You can also develop in .py files.
- Panel is one of the 4 most popular data app frameworks in Python with `more than 400.000 downloads a month <https://pyviz.org/tools.html#dashboarding>`_. It's especially popular in the scientific community.
- Panel is used by for example Rapids to power `CuxFilter`_, a CuDF based big data viz framework.
- Panel can be deployed on your favorite server or cloud including `Lightning`_.

Panel is **particularly well suited for lightning.ai apps** that needs to display live progress from
`LightningWork` as the Panel server can react to progress and asynchronously push messages from the server to the
client via web socket communication.

Install panel with:

.. code:: bash

    pip install panel

----

*************************
Create the Panel demo app
*************************

To explain how to use Panel with Lightning, let's build a simple app with Panel.

In the next few sections we'll build an app step-by-step.

First **create a file named app.py** with the app content:

.. code:: bash

    import lightning_app as lapp
    import panel as pn
    import plotly.express as px

    pn.extension("plotly", sizing_mode="stretch_width")

    ACCENT = "#792EE5"


    def get_panel_theme():
        """Returns 'default' or 'dark'"""
        return pn.state.session_args.get("theme", [b"default"])[0].decode()


    def get_plotly_template():
        if get_panel_theme() == "dark":
            return "plotly_dark"
        return "plotly_white"


    def get_plot(length=5):
        xseries = [index for index in range(length + 1)]
        yseries = [x**2 for x in xseries]
        fig = px.line(
            x=xseries,
            y=yseries,
            template=get_plotly_template(),
            color_discrete_sequence=[ACCENT],
            range_x=(0, 10),
            markers=True,
        )
        fig.layout.autosize = True
        return fig


    def get_view():
        length = pn.widgets.IntSlider(value=5, start=1, end=10, name="Length")
        plot = pn.bind(get_plot, length=length)
        component = pn.Column(length, plot)
        template = pn.template.FastListTemplate(
            title="⚡ Hello Panel + Lightning ⚡", main=[component], accent=ACCENT
        )
        return template


    class LitPanel(lapp.LightningWork):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.requests = 0

        def _fast_initial_view(self):
            self.requests += 1
            if self.requests == 1:
                return pn.pane.HTML("<h1>Please refresh the browser to see the app.</h1>")
            else:
                return get_view()

        def run(self):
            pn.serve(
                {"/": self._fast_initial_view},
                port=self.port,
                address=self.host,
                websocket_origin="*",
                show=False,
            )


    class LitApp(lapp.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_panel = LitPanel(parallel=True)

        def run(self):
            self.lit_panel.run()

        def configure_layout(self):
            tab1 = {"name": "Home", "content": self.lit_panel}
            return tab1


    app = lapp.LightningApp(LitApp())

    if __name__.startswith("bokeh"):
        LitPanel().view().servable()

        


Add `panel`, `lightning`, `pandas` and `plotly` to your requirements.txt file

.. code:: bash

    lightning
    panel
    pandas
    plotly

this is a best practice to make apps reproducible.

----

***********
Run the app
***********

Develop the panel app locally with hot reload

.. code:: bash

    panel serve app.py --autoreload --show

Test the lightning app locally

.. code:: bash

    lightning run app app.py

Now run it on the cloud as well:

.. code:: console

    lightning run app app.py --cloud

----

************************
Step-by-step walkthrough
************************

In this section, we explain each part of this code in detail.

----

1. Define a Panel app
^^^^^^^^^^^^^^^^^^^^^

First, find the Panel app you want to integrate. In this example, that app looks like:

.. code:: python

        import panel as pn
        import plotly.express as px

        pn.extension("plotly", sizing_mode="stretch_width")

        ACCENT = "#792EE5"


        def get_panel_theme():
            """Returns 'default' or 'dark'"""
            return pn.state.session_args.get("theme", [b"default"])[0].decode()


        def get_plotly_template():
            if get_panel_theme() == "dark":
                return "plotly_dark"
            return "plotly_white"


        def get_plot(length=5):
            xseries = [index for index in range(length + 1)]
            yseries = [x**2 for x in xseries]
            fig = px.line(
                x=xseries,
                y=yseries,
                template=get_plotly_template(),
                color_discrete_sequence=[ACCENT],
                range_x=(0, 10),
                markers=True,
            )
            fig.layout.autosize = True
            return fig


        def get_view():
            length = pn.widgets.IntSlider(value=5, start=1, end=10, name="Length")
            plot = pn.bind(get_plot, length=length)
            component = pn.Column(length, plot)
            template = pn.template.FastListTemplate(
                title="⚡ Hello Panel + Lightning ⚡", main=[component], accent=ACCENT
            )
            return template

        get_view().servable()

You can serve the app by running

.. code:: bash

    panel serve 'name_of_script.py' --autoreload --show

This Panel app plots a simple line curve depending on the value of a slider.
Its all wrapped in a nice and easy to use template.

`Visit the Panel documentation for the full API <https://panel.holoviz.org>`_.

----

2. Add the Panel app to a lightning component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add the Panel app to the run method of a ``LightningWork`` component and run the server on that
component's **host** and **port**.

.. code:: python
    :emphasize-lines: 7, 8, 10

    class LitPanel(lapp.LightningWork):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.requests = 0

        def _fast_initial_view(self):
            self.requests += 1
            if self.requests == 1:
                return pn.pane.HTML("<h1>Please refresh the browser to see the app.</h1>")
            else:
                return get_view()

        def run(self):
            pn.serve(
                {"/": self._fast_initial_view},
                port=self.port,
                address=self.host,
                websocket_origin="*",
                show=False,
            )

Please note we need to wrap the `view` function by the `_fast_initial_view` such that the first,
initial request done by the lightning server is fast. Otherwise it will continue requesting a
response and never load the page.

----

3. Route the UI in the root component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The final step, is to tell the Root component in which tab to render this component's UI.
In this case, we render the ``LitPanel`` UI in the ``Home`` tab of the application.

.. code:: python
    :emphasize-lines: 4, 7, 10

        class LitApp(lapp.LightningFlow):
            def __init__(self):
                super().__init__()
                self.lit_panel = LitPanel(parallel=True)

            def run(self):
                self.lit_panel.run()

            def configure_layout(self):
                tab1 = {"name": "Home", "content": self.lit_panel}
                return tab1

        app = lapp.LightningApp(LitApp())

We use the ``parallel=True`` argument of ``LightningWork`` to run the server in the background
while the rest of the Lightning App runs everything else.

4. Add `.servable()`` to enable fast development with hot reload
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To enable hot reload with `panel serve app.py --autoreload --show` we add

.. code:: python

    if __name__.startswith("bokeh"):
        LitPanel().view().servable()

.. _Panel: https://panel.holoviz.org/
.. _HoloViz: https://holoviz.org/
.. _DataShader: https://datashader.org/
.. _HoloViews: https://holoviews.org/
.. _Lightning: https://lightning.ai/
.. _CuxFilter: https://github.com/rapidsai/cuxfilter
