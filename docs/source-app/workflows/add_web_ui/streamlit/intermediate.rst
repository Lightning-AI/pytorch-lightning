##########################################
Add a web UI with Streamlit (intermediate)
##########################################
**Audience:** Users who want to communicate between the Lightning App and Streamlit.

**Prereqs:** Must have read the :doc:`streamlit basic <basic>` guide.

----

************************************
Interact with the App from Streamlit
************************************
The streamlit UI enables user interactions with the Lightning App via UI elements like buttons.
To modify the variables of a Lightning component, access the ``lightning_app_state`` variable in .

For example, here we increase the count variable of the Lightning Component every time a user presses a button:

.. code:: python
    :emphasize-lines: 8, 14

    # app.py
    import lightning as L
    import lightning.app.frontend as frontend
    import streamlit as st


    def your_streamlit_app(lightning_app_state):
        if st.button("press to increase count"):
            lightning_app_state.count += 1
        st.write(f"current count: {lightning_app_state.count}")


    class LitStreamlit(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.count = 0

        def configure_layout(self):
            return frontend.StreamlitFrontend(render_fn=your_streamlit_app)


    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_streamlit = LitStreamlit()

        def run(self):
            self.lit_streamlit.run()

        def configure_layout(self):
            tab1 = {"name": "home", "content": self.lit_streamlit}
            return tab1


    app = L.LightningApp(LitApp())

----

****************************************
Interact with Streamlit from a component
****************************************
To update the streamlit UI from any Lightning component, update the property in the component and make sure to call ``run`` from the
parent component.

In this example we update the value of the counter from the component:

.. code:: python
    :emphasize-lines: 7, 15

    # app.py
    import lightning as L
    import lightning.app.frontend as frontend
    import streamlit as st


    def your_streamlit_app(lightning_app_state):
        st.write(f"current count: {lightning_app_state.count}")


    class LitStreamlit(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.count = 0

        def run(self):
            self.count += 1

        def configure_layout(self):
            return frontend.StreamlitFrontend(render_fn=your_streamlit_app)


    class LitApp(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.lit_streamlit = LitStreamlit()

        def run(self):
            self.lit_streamlit.run()

        def configure_layout(self):
            tab1 = {"name": "home", "content": self.lit_streamlit}
            return tab1


    app = L.LightningApp(LitApp())
