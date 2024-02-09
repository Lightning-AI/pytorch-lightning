:orphan:

###############################
Add a web UI with Panel (basic)
###############################

**Audience:** Users who want to add a web UI written with Python and Panel.

**Prereqs:** Basic Python knowledge.

----

**************
What is Panel?
**************

`Panel`_ and the `HoloViz`_ ecosystem provide unique and powerful
features such as big data visualization using `DataShader`_, easy cross filtering
using `HoloViews`_, streaming and much more.

* Panel is highly flexible and ties into the PyData and Jupyter ecosystems as you can develop in notebooks and use ipywidgets. You can also develop in .py files.

* Panel is one of the most popular data app frameworks in Python with `more than 400.000 downloads a month <https://pyviz.org/tools.html#dashboarding>`_. It's especially popular in the scientific community.

* Panel is used, for example, by Rapids to power `CuxFilter`_, a CuDF based big data visualization framework.

* Panel can be deployed on your favorite server or cloud including `Lightning`_.

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/panel-intro.gif
   :alt: Example Panel App

   Example Panel App

Panel is **particularly well suited for Lightning Apps** that need to display live progress. This is because the Panel server can react
to state changes and asynchronously push messages from the server to the client using web socket communication.

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/panel-streaming-intro.gif
   :alt: Example Panel Streaming App

   Example Panel Streaming App

Install Panel with:

.. code:: bash

    pip install panel

----

*********************
Run a basic Panel App
*********************

In the next few sections, we'll build an App step-by-step.

First, create a file named ``app_panel.py`` with the App content:

.. code:: python

    # app_panel.py

    import panel as pn

    pn.panel("Hello **Panel ⚡** World").servable()

Then, create a file named ``app.py`` with the following App content:

.. code:: python

    # app.py

    import lightning as L
    from lightning.app.frontend import PanelFrontend


    class LitPanel(L.LightningFlow):

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

Finally, add ``panel`` to your ``requirements.txt`` file:

.. code:: bash

    echo 'panel' >> requirements.txt

.. note:: This is a best practice to make Apps reproducible.

----

***********
Run the App
***********

Run the App locally:

.. code:: bash

    lightning_app run app app.py

The App should look like this:

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/panel-lightning-basic.png
   :alt: Basic Panel Lightning App

   Basic Panel Lightning App

Now, run it on the cloud:

.. code:: bash

    lightning_app run app app.py --cloud

----

*************************
Step-by-step walk-through
*************************

In this section, we explain each part of the code in detail.

----

0. Define a Panel app
^^^^^^^^^^^^^^^^^^^^^

First, find the Panel app you want to integrate. In this example, that app looks like:

.. code:: python

    import panel as pn

    pn.panel("Hello **Panel ⚡** World").servable()

Refer to the `Panel documentation <https://panel.holoviz.org/>`_ and `awesome-panel <https://github.com/awesome-panel/awesome-panel>`_ for more complex examples.

----

1. Add Panel to a Component
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Link this app to the Lightning App by using the ``PanelFrontend`` class which needs to be returned from
the ``configure_layout`` method of the Lightning Component you want to connect to Panel.

.. code:: python
    :emphasize-lines: 7-10

    import lightning as L
    from lightning.app.frontend import PanelFrontend


    class LitPanel(L.LightningFlow):

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

The argument of the ``PanelFrontend`` class, points to the script, notebook, or function that
runs your Panel app.

----

2. Route the UI in the root component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The second step, is to tell the Root component in which tab to render this component's UI.
In this case, we render the ``LitPanel`` UI in the ``home`` tab of the app.

.. code:: python
    :emphasize-lines: 19-20

    import lightning as L
    from lightning.app.frontend import PanelFrontend


    class LitPanel(L.LightningFlow):

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

----

*************
Tips & Tricks
*************

0. Use autoreload while developing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To speed up your development workflow, you can run your Lightning App with Panel **autoreload** by
setting the environment variable ``PANEL_AUTORELOAD`` to ``yes``.

Try running the following:

.. code-block::

    PANEL_AUTORELOAD=yes lightning run app app.py

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/panel-lightning-autoreload.gif
   :alt: Basic Panel Lightning App with autoreload

   Basic Panel Lightning App with autoreload

1. Theme your App
^^^^^^^^^^^^^^^^^

To theme your App you, can use the Lightning accent color ``#792EE5`` with the `FastListTemplate`_.

Try replacing the contents of ``app_panel.py`` with the following:

.. code:: bash

    # app_panel.py

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


Install some additional libraries and remember to add the dependencies to the ``requirements.txt`` file:


.. code:: bash

    echo 'plotly' >> requirements.txt
    echo 'pandas' >> requirements.txt

Finally run the App

.. code:: bash

    lightning_app run app app.py

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/panel-lightning-theme.gif
   :alt: Basic Panel Plotly Lightning App with theming

   Basic Panel Plotly Lightning App with theming

.. _Panel: https://panel.holoviz.org/
.. _FastListTemplate: https://panel.holoviz.org/reference/templates/FastListTemplate.html#templates-gallery-fastlisttemplate
.. _HoloViz: https://holoviz.org/
.. _DataShader: https://datashader.org/
.. _HoloViews: https://holoviews.org/
.. _Lightning: https://lightning.ai/
.. _CuxFilter: https://github.com/rapidsai/cuxfilter
.. _AwesomePanel: https://github.com/awesome-panel/awesome-panel


----

**********
Next Steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: 2: Enable two-way communication
   :description: Enable two-way communication between Panel and a Lightning App.
   :col_css: col-md-6
   :button_link: intermediate.html
   :height: 150
   :tag: intermediate

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
