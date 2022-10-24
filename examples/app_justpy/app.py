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


def render_fn(get_state: Callable) -> Callable:
    import justpy as jp

    def webpage():
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

    return webpage


app = LightningApp(Flow())
