:orphan:

########################
Add a web UI with JustPy
########################


******
JustPy
******

The framework `JustPy <https://github.com/justpy-org/justpy>`_  is an object oriented high-level Python Web Framework that requires no frontend programming.

Additionally, it provides a higher level API called `Quasar <https://justpy.io/quasar_tutorial/introduction/>`_ with stylized components.


You can install ``justpy`` from PyPi.

.. code-block::

    pip install justpy

*******
Example
*******


In the following example, we are creating a simple UI with 2 buttons.
When clicking the first button, the flow state ``counter`` is incremented and re-rendered on the UI.


First of all, you would need to import the ``JustPyFrontend`` and return it from the ``configure_layout`` hook of the flow.

.. code-block::

    from typing import Callable

    from lightning import LightningApp, LightningFlow
    from lightning.app.frontend import JustPyFrontend


    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.counter = 0

        def run(self):
            print(self.counter)

        def configure_layout(self):
            return JustPyFrontend(render_fn=render_fn)

Secondly, you would need to implement a ``render_fn`` that takes as input a ``get_state`` function and return a function.


.. code-block::

    def render_fn(get_state: Callable) -> Callable:
        import justpy as jp

        def website():
            wp = jp.QuasarPage(dark=True)
            d = jp.Div(classes="q-pa-md q-gutter-sm", a=wp)
            container = jp.QBtn(color="primary", text="Counter: 0")

            async def click(*_):
                state = get_state()
                state.counter += 1
                container.text = f"Counter: {state.counter}"

            button = jp.QBtn(color="primary", text="Click Me!", click=click)

            d.add(button)
            d.add(container)

            return wp

        return website


Finally, you can simply run your app.

.. code-block::

    app = LightningApp(Flow())
