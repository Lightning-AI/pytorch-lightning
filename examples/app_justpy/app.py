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

    async def my_click(self, msg):
        state = get_state()
        state.counter += 1
        msg.page.components[0].components[1].text = f"Counter: {state.counter}"

    def website():
        wp = jp.QuasarPage(dark=True)
        d = jp.Div(classes="q-pa-md q-gutter-sm", a=wp)
        jp.QBtn(color="primary", text="Click Me!", click=my_click, a=d)
        jp.QBtn(color="primary", text="Counter: 0", a=d)
        return wp

    return website


app = LightningApp(Flow())
