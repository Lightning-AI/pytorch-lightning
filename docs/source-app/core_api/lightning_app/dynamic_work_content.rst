***************************************
What Dynamic LightningWork does for you
***************************************

Dynamic LightningWork (Work) changes the resources your application uses while the application is running (aka at runtime).

For example, imagine you want to create a research notebook app for your team. You want every member to be able to create multiple `JupyterLab <https://jupyter.org/>`_ sessions on their hardware of choice.

To allow every notebook to choose hardware, it needs to be set up in it's own :class:`~lightning.app.core.work.LightningWork`, but you can't know the number of notebooks user will need in advance. In this case you'll need to add ``LightningWorks`` dynamically at run time.

----

*****************
Use Dynamic Works
*****************

Dynamic Works should be used anytime you want change the resources your application is using while it is running (aka at runtime).

You're usually going to use the ``start`` and ``stop`` methods together.

----

Add a Dynamic Work
^^^^^^^^^^^^^^^^^^

There are a couple of ways you can add a dynamic Work:

- Option 1: Attach your components in the **run** method using the Python functions.
- Option 2: Use the Lightning built-in classes :class:`~lightning.app.structures.Dict` or :class:`~lightning.app.structures.List`.

.. note:: Using the Lightning built-in classes is usually easier to read.

----

**OPTION 1:** Attach your components in the run method of a flow using the Python functions **hasattr**, **setattr**, and **getattr**:

.. code-block:: python

    class RootFlow(lapp.LightningFlow):

        def run(self):

            if not hasattr(self, "work"):
                # The `Work` component is created and attached here.
                setattr(self, "work", Work())
            # Run the `Work` component.
            getattr(self,  "work").run()

**OPTION 2:** Use the built-in Lightning classes :class:`~lightning.app.structures.Dict` or :class:`~lightning.app.structures.List`

.. code-block:: python

    from lightning.app.structures import Dict

    class RootFlow(lapp.LightningFlow):

        def __init__(self):
            super().__init__()
            self.dict = Dict()

        def run(self):
            if "work" not in self.dict:
                # The `Work` component is attached here.
                self.dict["work"] = Work()
            self.dict["work"].run()

----

Stop a Work
^^^^^^^^^^^
Stop a work when you are concerned about cost.

To stop a work, use the work ``stop`` method:

.. code-block:: python

    class RootFlow(L.LightningFlow):

        def __init__(self):
            super().__init__()
            self.work = Work()

        def run(self):
            self.work.stop()

----

*********************
Dynamic Work Examples
*********************

..
    The entire application can be found `here <https://github.com/Lightning-AI/lightning-template-jupyterlab>`_.

----

Dynamic Work with Jupyter Notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we are dynamically creating ``JupyterLabWork`` every time a user clicks the **Create Jupyter Notebook** button.

In order to do that, we are iterating over the list of ``jupyter_config_requests`` infinitely.

.. code-block:: python

    import lightning as L


    class JupyterLabManager(L.LightningFlow):

        """This flow manages the users notebooks running within works.""""

        def __init__(self):
            super().__init__()
            self.jupyter_works = L.structures.Dict()
            self.jupyter_config_requests = []

        def run(self):
            for idx, jupyter_config in enumerate(self.jupyter_config_requests):

                # The Jupyter Config has this form is:
                # {"use_gpu": False/True, "token": None, "username": ..., "stop": False}

                # Step 1: Check if JupyterWork already exists for this username
                username = jupyter_config["username"]
                if username not in self.jupyter_works:
                    jupyter_config["ready"] = False

                    # Set the hardware selected by the user: GPU or CPU.
                    cloud_compute = L.CloudCompute("gpu" if jupyter_config["use_gpu"] else "cpu-small")

                    # Step 2: Create new JupyterWork dynamically !
                    self.jupyter_works[username] = JupyterLabWork(cloud_compute=cloud_compute)

                # Step 3: Run the JupyterWork
                self.jupyter_works[username].run()

                # Step 4: Store the notebook token in the associated config.
                # We are using this to know when the notebook is ready
                # and display the stop button on the UI.
                if self.jupyter_works[username].token:
                    jupyter_config["token"] = self.jupyter_works[username].token

                # Step 5: Stop the work if the user requested it.
                if jupyter_config['stop']:
                    self.jupyter_works[username].stop()
                    self.jupyter_config_requests.pop(idx)

        def configure_layout(self):
            return L.app.frontend.StreamlitFrontend(render_fn=render_fn)

----

Dynamic Works with StreamLit UI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Continuing from the Jupyter Notebook example, in the UI, we receive the **state** of the Jupyter Manager and the state can be modified directly from the UI.

.. code-block:: python

    import streamlit as st


    def render_fn(state):

        # Step 1: Enable users to select their notebooks and create them
        column_1, column_2, column_3 = st.columns(3)
        with column_1:
            create_jupyter = st.button("Create Jupyter Notebook")
        with column_2:
            username = st.text_input('Enter your username', "tchaton")
            assert username
        with column_3:
            use_gpu = st.checkbox('Use GPU')

        # Step 2: If a user clicked the button, add an element to the list of configs
        # Note: state.jupyter_config_requests = ... will sent the state update to the component.
        if create_jupyter:
            new_config = [{"use_gpu": use_gpu, "token": None, "username": username, "stop": False}]
            state.jupyter_config_requests = state.jupyter_config_requests + new_config

        # Step 3: List of running notebooks.
        for idx, config in enumerate(state.jupyter_config_requests):
            column_1, column_2, column_3 = st.columns(3)
            with column_1:
                if not idx:
                    st.write(f"Idx")
                st.write(f"{idx}")
            with column_2:
                if not idx:
                    st.write(f"Use GPU")
                st.write(config['use_gpu'])
            with column_3:
                if not idx:
                    st.write(f"Stop")
                if config["token"]:
                    should_stop = st.button("Stop this notebook")

                    # Step 4: Change stop if the user clicked the button
                    if should_stop:
                        config["stop"] = should_stop
                        state.jupyter_config_requests = state.jupyter_config_requests
