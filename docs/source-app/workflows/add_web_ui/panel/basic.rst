###################################
Add a web UI with Panel (basic)
###################################

**Audience:** Users who want to add a web UI written with Python.

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

.. figure:: https://cdn.jsdelivr.net/gh/MarcSkovMadsen/awesome-panel-assets@master/videos/panel-lightning/panel-intro.gif
   :alt: Example Panel App

   Example Panel App

Panel is **particularly well suited for lightning.ai apps** that needs to display live progress from
`LightningWork` as the Panel server can react to progress and asynchronously push messages from the server to the
client via web socket communication.

.. figure:: https://cdn.jsdelivr.net/gh/MarcSkovMadsen/awesome-panel-assets@master/videos/panel-lightning/panel-streaming.gif
   :alt: Example Panel Streaming App

   Example Panel Streaming App

Install Panel with:

.. code:: bash

    pip install panel

----

*************************
Run a basic Panel app
*************************

In the next few sections we'll build an app step-by-step.

First **create a file named ``panel_app_basic.py``** with the app content:

.. code:: python

    import panel as pn

    pn.panel("Hello **Panel ⚡** World").servable()

Then **create a file named ``app_basic.py``** with the app content:

.. code:: python

    import lightning as L
    from lightning_app.frontend.panel import PanelFrontend

    class LitPanel(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self._frontend = PanelFrontend("panel_app_basic.py")

        def configure_layout(self):
            return self._frontend

    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_panel = LitPanel()

        def configure_layout(self):
            return {"name": "home", "content": self.lit_panel}


    app = L.LightningApp(LitApp())

add "panel" to a requirements.txt file:

.. code:: bash

    echo 'panel' >> requirements.txt

this is a best practice to make apps reproducible.

----

***********
Run the app
***********

Run the app locally to see it!

.. code:: bash

    lightning run app app_basic.py

.. figure:: https://cdn.jsdelivr.net/gh/MarcSkovMadsen/awesome-panel-assets@master/images/panel-lightning/panel-lightning-basic.png
   :alt: Basic Panel Lightning App

   Basic Panel Lightning App

Now run it on the cloud as well:

.. code:: bash

    lightning run app app_basic.py --cloud

----

************************
Step-by-step walkthrough
************************

In this section, we explain each part of this code in detail.

----

0. Define a Panel app
^^^^^^^^^^^^^^^^^^^^^^^^^

First, find the Panel app you want to integrate. In this example, that app looks like:

.. code:: python

    import panel as pn

    pn.panel("Hello **Panel ⚡** World").servable()

Refer to the `Panel documentation <https://docs.Panel.io/>`_ or `awesome-panel.org <https://awesome-panel.org>`_ for more complex examples.

----

1. Add Panel to a component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Link this app to the Lightning App by using the ``PanelFrontend`` class which needs to be returned from
the ``configure_layout`` method of the Lightning component you want to connect to Panel.

.. code:: python
    :emphasize-lines: 7,10

    import lightning as L
    from lightning_app.frontend.panel import PanelFrontend

    class LitPanel(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self._frontend = PanelFrontend("panel_app_basic.py")

        def configure_layout(self):
            return self._frontend

    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_panel = LitPanel()

        def configure_layout(self):
            return {"name": "home", "content": self.lit_panel}


    app = L.LightningApp(LitApp())

The argument of the ``PanelFrontend`` class, points to the script, notebook or function that
runs your Panel app.

----

2. Route the UI in the root component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The second step, is to tell the Root component in which tab to render this component's UI.
In this case, we render the ``LitPanel`` UI in the ``home`` tab of the application.

.. code:: python
    :emphasize-lines: 18

    import lightning as L
    from lightning_app.frontend.panel import PanelFrontend

    class LitPanel(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self._frontend = PanelFrontend("panel_app_basic.py")

        def configure_layout(self):
            return self._frontend

    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_panel = LitPanel()

        def configure_layout(self):
            return {"name": "home", "content": self.lit_panel}

**********
Autoreload
**********

You can run your lightning app with Panel **autoreload** by setting the environment variable
``PANEL_AUTORELOAD`` to ``yes``.

.. code-block::

    PANEL_AUTORELOAD=yes lightning run app app_basic.py

.. figure:: https://cdn.jsdelivr.net/gh/MarcSkovMadsen/awesome-panel-assets@master/videos/panel-lightning/panel-lightning-autoreload.gif
   :alt: Basic Panel Lightning App with autoreload

   Basic Panel Lightning App with autoreload

*******
Theming
*******

To theme your app you, can use the lightning accent color #792EE5 with the `FastListTemplate`_.

Try replacing the contents of ``app_basic.py`` with the below code.

.. code:: bash

    import panel as pn
    import plotly.express as px

    ACCENT = "#792EE5"

    pn.extension("plotly", sizing_mode="stretch_width", template="fast")
    pn.state.template.param.update(
        title="⚡ Hello Panel + Lightning ⚡", accent_base_color=ACCENT, header_background=ACCENT
    )

    pn.config.raw_css.append(
        """
    .bk-root:first-of-type {
        height: calc( 100vh - 200px ) !important;
    }
    """
    )


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


    length = pn.widgets.IntSlider(value=5, start=1, end=10, name="Length")
    dynamic_plot = pn.panel(
        pn.bind(get_plot, length=length), sizing_mode="stretch_both", config={"responsive": True}
    )
    pn.Column(length, dynamic_plot).servable()

Run `pip install plotly pandas` and remember to add the dependencies to the requirements.txt file:

.. code:: bash

    echo 'plotly' >> requirements.txt
    echo 'pandas' >> requirements.txt

Finally run the app

.. code:: bash

    lightning run app app_basic.py

.. figure:: https://cdn.jsdelivr.net/gh/MarcSkovMadsen/awesome-panel-assets@master/videos/panel-lightning/panel-lightning-theme.gif
   :alt: Basic Panel Plotly Lightning App with theming

   Basic Panel Plotly Lightning App with theming

.. _Panel: https://panel.holoviz.org/
.. _FastListTemplate: https://panel.holoviz.org/reference/templates/FastListTemplate.html#templates-gallery-fastlisttemplate
.. _HoloViz: https://holoviz.org/
.. _DataShader: https://datashader.org/
.. _HoloViews: https://holoviews.org/
.. _Lightning: https://lightning.ai/
.. _CuxFilter: https://github.com/rapidsai/cuxfilter
.. _AwesomePanel: https://awesome-panel.org/home
